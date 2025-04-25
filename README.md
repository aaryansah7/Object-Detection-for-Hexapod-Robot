Real-Time Object Detection & Audio Feedback using YOLOv8
This project showcases a real-time object detection system powered by the YOLOv8 model from Ultralytics, integrated with OpenCV for video processing and gTTS + pygame for generating speech-based feedback. When an object is detected through your webcam, the system not only draws bounding boxes and labels but also announces the object name aloud the first time it's seen.

🧠 Key Features:
- Uses YOLOv8s pretrained model for high-speed and accurate object detection.
- Real-time detection from your webcam feed using OpenCV.
- Integrates Text-to-Speech (TTS) to vocalize object names when detected.
- Avoids repeating object names using a smart detection cache.
- Displays detection confidence and FPS in the video window.

📦 Technologies Used:
- Python
- OpenCV
- Ultralytics YOLOv8
- gTTS (Google Text-to-Speech)
- pygame
- threading

#### Perfect for use cases like accessibility tools, smart surveillance, or interactive AI demos.
