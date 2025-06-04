import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
import speech_recognition as sr
import cv2
import numpy as np
import pyttsx3
import time
import threading
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slower speech rate for clarity

# Safe text-to-speech using threading
def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()
    st.session_state.last_spoken = text

# Load YOLOv8 model for obstacle detection
model = YOLO("D:/AI Project/yolov8_finetuned.pt")  # Update with your model path

# Function to detect obstacles
def detect_obstacles(image):
    results = model(image)
    detected_obstacles = set()
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.3:
                detected_obstacles.add(label)
    return list(detected_obstacles)

# Generate navigation guidance using Gemini Pro
def generate_guidance(detected_obstacles, user_query, input_type="voice"):
    obstacle_text = ", ".join(detected_obstacles) if detected_obstacles else "no significant obstacles"
    prompt = f"""
    You are a helpful navigation assistant for visually impaired individuals. 
    Provide clear, concise guidance with these rules:
    1. Begin with a brief summary of detected obstacles
    2. Provide specific guidance based on the user's query
    3. Include relative positions (left/right/center, near/far)
    4. Warn about immediate dangers first
    5. Suggest safe paths using clock directions when possible
    6. Keep responses under 25 words for voice and 40 words for text
    7. Use simple, direct language

    Current environment contains: {obstacle_text}
    User {'asked' if input_type == 'voice' else 'typed'}: "{user_query}"
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Guidance unavailable. Please try again. Error: {str(e)}"

# Voice-to-text
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.session_state.listening = True
        st.info("🎤 Listening... Speak now")
        audio = recognizer.listen(source, timeout=5)
        st.info("🎙 Processing your request...")
        try:
            text = recognizer.recognize_google(audio)
            st.session_state.listening = False
            return text
        except sr.UnknownValueError:
            st.session_state.listening = False
            return "Could not understand audio"
        except sr.RequestError:
            st.session_state.listening = False
            return "Speech service unavailable"
        except Exception as e:
            st.session_state.listening = False
            return f"Error: {str(e)}"

# Streamlit UI
st.title("Visually Impaired Navigation Assistant 🦯")
st.write("Real-time obstacle detection and audio guidance system")

if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = ""
if 'last_guidance' not in st.session_state:
    st.session_state.last_guidance = ""

upload_option = st.radio("Input method:", ["Upload image", "Use device camera"], horizontal=True)

img = None
if upload_option == "Upload image":
    uploaded_file = st.file_uploader("Upload environment image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
else:
    picture = st.camera_input("Take a picture of your surroundings")
    if picture:
        img = Image.open(picture)

if img is not None:
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    detected_obstacles = detect_obstacles(img_array)
    
    st.subheader("Detection Results")
    if detected_obstacles:
        detection_text = "I detected: " + ", ".join(detected_obstacles)
        st.write(detection_text)
        speak(detection_text)
    else:
        st.write("No significant obstacles detected")
        speak("No significant obstacles detected")
    
    results = model(img_array)
    rendered_img = results[0].plot()
    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
    st.image(rendered_img, caption="Obstacle Detection", use_column_width=True)
    
    st.subheader("Navigation Assistance")
    
    input_method = st.radio("Query method:", ["Voice command", "Text input"], index=0, horizontal=True)
    query = ""
    if input_method == "Voice command":
        if st.button("🎤 Press and speak"):
            query = record_audio()
            if query not in ["Could not understand audio", "Speech service unavailable"]:
                st.write(f"You said: {query}")
    else:
        query = st.text_input("Type your navigation question:", placeholder="e.g., 'What's in front of me?'")
    
    if query and query not in ["Could not understand audio", "Speech service unavailable"]:
        with st.spinner("Generating guidance..."):
            guidance = generate_guidance(detected_obstacles, query, "voice" if input_method == "Voice command" else "text")
            st.subheader("Navigation Guidance")
            st.write(guidance)
            speak(guidance)
            st.session_state.last_guidance = guidance
    
    if st.session_state.last_guidance:
        if st.button("🔊 Repeat last guidance"):
            speak(st.session_state.last_guidance)
    
    st.subheader("Quick Commands")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What's around me?"):
            guidance = generate_guidance(detected_obstacles, "Describe all obstacles around me with their positions")
            st.write(guidance)
            speak(guidance)
    with col2:
        if st.button("Find safe path"):
            guidance = generate_guidance(detected_obstacles, "Suggest a safe path to move forward avoiding all obstacles")
            st.write(guidance)
            speak(guidance)
    with col3:
        if st.button("Any dangers?"):
            guidance = generate_guidance(detected_obstacles, "What are the most dangerous obstacles I should avoid immediately?")
            st.write(guidance)
            speak(guidance)

with st.sidebar:
    st.header("How to Use")
    st.write("""
    1. Capture/upload environment image
    2. System announces detected obstacles
    3. Ask questions using voice or text
    4. Listen to audio guidance
    5. Use quick commands
    """)
    st.header("Key Features")
    st.write("""
    - Real-time obstacle detection
    - Voice & text interaction
    - Accessible feedback
    - Quick commands
    """)
    st.header("Accessibility Options")
    tts_rate = st.slider("Speech Rate", 100, 300, 150)
    engine.setProperty('rate', tts_rate)
    if st.button("Test Audio"):
        speak("This is a test of the audio guidance system")