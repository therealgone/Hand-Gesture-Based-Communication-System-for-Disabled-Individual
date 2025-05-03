from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import threading
import asyncio
from datetime import datetime
import numpy as np
from bleak import BleakClient
import time
import subprocess
import platform
import socket

app = Flask(__name__)

# ===== Global Variables =====
sentence = []
latest_word = ""
final_response = ""
bluetooth_mac = None
bluetooth_client = None
bluetooth_connected = False
log_file = "logs.json"
presets_file = "presets.json"
temp_speech_file = "temp_speech.txt"

# Try to load gesture labels to get preset names
try:
    gesture_labels = np.load("preset_gesture_labels.npy")
    preset_keys = [label for label in gesture_labels if label.startswith("preset")]
except Exception as e:
    print(f"Could not load gesture_labels.npy: {e}")
    preset_keys = ["preset1", "preset2", "preset3"]

# Load or create presets file
if not os.path.exists(presets_file):
    # Initialize with default values
    presets = {}
    for key in preset_keys:
        presets[key] = key.replace("preset", "preset ")  # Default values
    
    with open(presets_file, 'w') as f:
        json.dump(presets, f)
else:
    with open(presets_file, 'r') as f:
        presets = json.load(f)
    
    # Make sure all preset keys from gesture_labels are in the presets file
    for key in preset_keys:
        if key not in presets:
            presets[key] = key.replace("preset", "preset ")  # Default values
    
    # Save any new presets
    with open(presets_file, 'w') as f:
        json.dump(presets, f)

print("Loaded presets:", presets)

# ===== Bluetooth Methods =====
async def connect_bluetooth(mac):
    global bluetooth_client, bluetooth_connected
    try:
        client = BleakClient(mac)
        await client.connect()
        bluetooth_client = client
        bluetooth_connected = True
        return True
    except Exception as e:
        print("Bluetooth connection error:", e)
        bluetooth_connected = False
        return False

async def send_to_bluetooth(text):
    global bluetooth_client
    if bluetooth_client and bluetooth_connected:
        try:
            await bluetooth_client.write_gatt_char("0000ffe1-0000-1000-8000-00805f9b34fb", text.encode())
            return True
        except Exception as e:
            print("Failed to send data:", e)
            return False
    return False

# ===== Helper Methods =====
def save_log(sentence, reply):
    log = {"timestamp": datetime.now().isoformat(), "sentence": sentence, "response": reply}
    
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    
    logs.append(log)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

# Greatly improved text-to-speech function using platform-specific methods and temporary files
def speak_text(text):
    # Save text to a temporary file first to ensure complete processing
    with open(temp_speech_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    system = platform.system()
    print(f"Speaking text via {system} system TTS: '{text}'")
    
    try:
        if system == 'Windows':
            # Use PowerShell (better than cmd.exe) for text-to-speech
            ps_command = (
                f'powershell -Command "'
                f'$text = Get-Content -Path \'{os.path.abspath(temp_speech_file)}\' -Raw; '
                f'Add-Type -AssemblyName System.Speech; '
                f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$speak.Rate = -2; ' # Slower
                f'$speak.Volume = 100; '
                f'$speak.Speak($text);"'
            )
            subprocess.run(ps_command, shell=True, check=True)
            
        elif system == 'Darwin':  # macOS
            # Use the 'say' command with the file
            subprocess.run(['say', '-f', temp_speech_file, '-r', '120'], check=True)
            
        elif system == 'Linux':
            # Try different TTS engines
            try:
                # Try espeak first
                subprocess.run(['espeak', '-f', temp_speech_file, '-s', '120', '-a', '200'], check=True)
            except FileNotFoundError:
                try:
                    # Try festival
                    subprocess.run(['festival', '--tts', temp_speech_file], check=True)
                except FileNotFoundError:
                    # If all else fails, try pyttsx3
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 120)
                        with open(temp_speech_file, 'r', encoding='utf-8') as f:
                            file_text = f.read()
                        engine.say(file_text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"All TTS methods failed: {e}")
        else:
            # Unknown system, try pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 120)
                with open(temp_speech_file, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                engine.say(file_text)
                engine.runAndWait()
            except Exception as e:
                print(f"pyttsx3 TTS fallback failed: {e}")
        
        print("Speech completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in TTS: {e}")
        # Fallback to pyttsx3 if other methods fail
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 120)
            with open(temp_speech_file, 'r', encoding='utf-8') as f:
                file_text = f.read()
            engine.say(file_text)
            engine.runAndWait()
            print("Speech completed with fallback")
            return True
        except Exception as e2:
            print(f"Fallback TTS also failed: {e2}")
            return False
    finally:
        # Clean up - optional, can keep for debugging
        if os.path.exists(temp_speech_file):
            os.remove(temp_speech_file)

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html", 
                          word=latest_word, 
                          sentence=" ".join(sentence), 
                          reply=final_response, 
                          connected=bluetooth_connected)

@app.route("/add", methods=["POST"])
def add_word():
    global sentence
    word = request.json.get("word")
    if word and word.strip():  # Only add non-empty words
        sentence.append(word)
    return jsonify({"sentence": sentence})

@app.route("/reset", methods=["POST"])
def reset_sentence():
    global sentence, final_response
    sentence = []
    final_response = ""
    return jsonify({"status": "cleared"})

@app.route("/gemini", methods=["POST"])
def run_gemini():
    global final_response, sentence
    
    if not sentence:
        return jsonify({"response": "No words in sentence."})
    
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    
    # Improved prompt for better sentence creation
    prompt = f"""
    Create a natural, conversational English sentence using these words: {' '.join(sentence)}
    
    Guidelines:
    - Make the sentence sound like something a real person would say
    - For short inputs (2-3 words), create a simple, everyday sentence
    - For longer inputs, create a more detailed sentence
    - Use natural language patterns and common expressions
    - If the words don't make a logical sentence, connect them in the most natural way possible
    - Don't add explanations or additional context
    - Return only the sentence
    - Make it sound friendly and conversational
    """
    
    response = model.generate_content(prompt)
    final_response = response.text.strip().replace("*", "").split("\n")[0]
    
    # Save log
    save_log(" ".join(sentence), final_response)
    
    # Send to Bluetooth if connected
    if bluetooth_connected:
        threading.Thread(target=lambda: asyncio.run(send_to_bluetooth(final_response))).start()
    
    # Speak the final response using text-to-speech in a separate thread
    speech_thread = threading.Thread(target=lambda: speak_text(final_response), daemon=False)
    speech_thread.start()
    
    return jsonify({"response": final_response, "audio": "playing"})

@app.route("/speak", methods=["POST"])
def speak_response():
    """Additional endpoint to manually trigger speech"""
    text = request.json.get("text", "")
    if text:
        threading.Thread(target=lambda: speak_text(text), daemon=False).start()
        return jsonify({"status": "speaking"})
    return jsonify({"status": "error", "message": "No text provided"})

@app.route("/connect", methods=["POST"])
def connect_bt():
    global bluetooth_mac
    bluetooth_mac = request.json.get("mac")
    
    def async_connect():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(connect_bluetooth(bluetooth_mac))
        return result
    
    # Start connection in background thread
    t = threading.Thread(target=async_connect)
    t.start()
    
    return jsonify({"status": "connecting"})

@app.route("/latest", methods=["GET"])
def latest():
    return jsonify({
        "word": latest_word, 
        "sentence": sentence,
        "connected": bluetooth_connected
    })

@app.route("/update_word", methods=["POST"])
def update_word():
    global latest_word
    latest_word = request.json.get("word", "")
    return jsonify({"status": "updated"})

@app.route("/logs", methods=["GET"])
def get_logs():
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
                return jsonify(logs)
            except json.JSONDecodeError:
                return jsonify([])
    return jsonify([])

@app.route("/presets", methods=["GET"])
def get_presets():
    return jsonify(presets)

@app.route("/update_preset", methods=["POST"])
def update_preset():
    global presets
    preset_key = request.json.get("key")
    preset_value = request.json.get("value")
    
    if preset_key and preset_value:
        presets[preset_key] = preset_value
        
        # Save to file
        with open(presets_file, "w") as f:
            json.dump(presets, f, indent=2)
        
        # Create a notification file to signal test.py to reload presets
        with open("preset_update.flag", "w") as f:
            f.write(str(time.time()))
        
        return jsonify({"status": "updated", "presets": presets})
    
    return jsonify({"status": "error", "message": "Missing key or value"})

@app.route("/logs_file", methods=["GET"])
def download_logs():
    if os.path.exists(log_file):
        return send_file(log_file, as_attachment=True)
    return "No logs found", 404

# Add a test endpoint to directly trigger speech without any other processing
@app.route("/test_speech", methods=["GET"])
def test_speech():
    test_text = "This is a test of the speech system. If you can hear this complete sentence, the system is working correctly."
    threading.Thread(target=lambda: speak_text(test_text), daemon=False).start()
    return jsonify({"status": "Test speech initiated", "text": test_text})

if __name__ == "__main__":
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\nServer will be accessible at:")
    print(f"Local: http://localhost:5000")
    print(f"Network: http://{local_ip}:5000")
    print("\nMake sure your phone is connected to the same WiFi network as this computer")
    
    # Run the Flask app with host='0.0.0.0' to allow external access
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)