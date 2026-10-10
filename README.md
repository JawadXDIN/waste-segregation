# Real-Time Waste Segregation System

College mini project: Real-Time Waste Segregation System Using Deep Learning and Computer Vision.

## Stack
Python + OpenCV + Ultralytics YOLO. No ChatGPT/OpenAI/Gemini API.

## Run
1. Install Python 3.10/3.11.
2. Open this folder in VS Code.
3. Terminal:
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Run:
   python app.py
5. Press Q to close the webcam.

If models/best.pt is absent, the app automatically runs a DEMO using a pretrained YOLO model. This proves the camera/inference pipeline but is NOT the final waste model.

## Final project
Train a custom waste model using the included train_waste_model_colab.ipynb.
Download best.pt and place it at:
models/best.pt

Recommended final classes:
plastic, paper, metal, glass, organic

Dataset must contain real YOLO-format bounding-box labels and have a license that permits your use.
