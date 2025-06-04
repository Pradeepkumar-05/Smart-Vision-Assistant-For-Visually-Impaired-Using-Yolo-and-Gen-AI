# Smart-Vision-Assistant-For-Visually-Impaired-Using-Yolo-and-Gen-AI

**Implementation :**

**1. Introduction :**

This project is a Streamlit-based multimodal assistant designed to assist visually impaired individuals in identifying and understanding objects in their surroundings. By combining YOLOv8 for object detection and Google Gemini for conversational voice assistance, the system provides real-time, voice-guided navigation support. It aims to enhance user safety, independence, and confidence in day-to-day environments like roads, bus stops, and crowded areas.

**2. Libraries Used :**
     • ultralytics – for YOLOv8 object detection

     • torch – PyTorch framework for deep learning operations

     • opencv-python – for image processing and camera support

     • streamlit – for creating the interactive web app

     • gtts / pyttsx3 – for converting text responses into voice

     • google.generativeai (Gemini) – for AI-powered conversational support

     • PIL – Python Imaging Library for image handling

**3. Features :**

     • Real-time object detection using YOLOv8

     • Voice-based interaction powered by Google Gemini

     • Text-to-speech responses for visually impaired users

     • Upload and analyze images from the environment

     • Object descriptions and directional cues (e.g., “bus is on the left”)

     • Lightweight and accessible Streamlit interface

     • Extensible for wearable device integration in future

**4. User Interface :**
The application is built with Streamlit, offering a clean and accessible layout. Users can upload an image or access a live camera feed. Once the image is processed, detected objects are highlighted with bounding boxes and labels. A voice assistant reads out the detected objects and provides additional context upon user request. The interface is designed to be minimalistic, responsive, and suitable for screen readers or voice navigation tools.

**5. Conclusion :**
This project demonstrates the effective use of AI and computer vision to support visually impaired individuals. By merging object detection with intelligent voice assistance, the system enables safer and more informed navigation in real-world environments. With future scope for video stream analysis, GPS guidance, and wearable device compatibility, the application sets a strong foundation for building inclusive and adaptive assistive technologies.
