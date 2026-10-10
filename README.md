# ♻️ Real-Time AI Waste Segregation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask&logoColor=black)
![Ultralytics](https://img.shields.io/badge/YOLO--World-Zero--Shot%20AI-orange)

An intelligent, real-time computer vision application that detects and categorizes waste into **Dry (Recyclable)**, **Wet (Organic)**, **E-Waste**, and **Hazardous** materials using cutting-edge Zero-Shot Object Detection (YOLO-World). 

This project aims to automate waste sorting at the source, promoting smarter recycling and sustainable waste management.

---

## ✨ Features
* **Zero-Shot AI Detection:** Powered by `YOLO-World`, enabling the detection of 35+ specific waste items (like *smartphones, earbuds, glass jars, apple, syringe*) **without needing custom dataset training**.
* **Real-time Web UI:** Beautiful, responsive, glowing Neumorphic dashboard built with Flask and CSS3.
* **Instant Categorization:** Dynamically maps scanned objects to appropriate bins (Dry ♻, Wet 🟢, E-Waste 🔋, Hazardous ⚠️).
* **Hardware Accelerated:** Optimized camera threading and DirectShow backend (`cv2.CAP_DSHOW`) for smooth, high-FPS webcam streaming on Windows.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** PyTorch, Ultralytics YOLO-World
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10 or 3.11** installed on your system. 

### 2. Clone / Setup the Repository
Open a terminal in the project directory and create a virtual environment:

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Usage

1. Start the Flask Server:
```bash
python app.py
```
2. Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
3. Allow browser camera permissions if prompted (or ensure your physical webcam is connected).
4. Hold any piece of waste (e.g., a plastic bottle, an apple, or a charging cable) in front of the camera, and the AI will categorize it in real-time!

---

## 🧠 Model Architecture & Zero-Shot Capabilities

Unlike traditional object detection pipelines that require massive, pre-annotated datasets for every new object, this project utilizes **YOLO-World**. 

YOLO-World uses Vision-Language representations (similar to CLIP) to understand physical objects based purely on their textual names. You can easily add new waste items to the scanner simply by typing their name in the `CUSTOM_CLASSES` array inside `app.py`. No training required!

### Supported Categories Out-of-the-Box:
- **Dry Waste:** Plastic bottle, Paper, Cardboard, Pen, Metal can, Glass, Drinking glass, Glass jar, Plastic bag, Plastic container.
- **Wet Waste:** Apple, Banana, Mosambi, Orange, Tomato, Cucumber, Vegetable, Fruit, Food waste.
- **E-Waste:** Phone, Smartwatch, Laptop, Battery, Earbuds, Airpods, Earphones, Electronic cable, Charger.
- **Hazardous Waste:** Syringe, Medicine blister pack.

---

## 📂 Project Structure

```text
├── app.py                # Main Flask server and YOLO inference engine
├── requirements.txt      # Python dependencies
├── DATASETS_AND_REFS.md  # Information on optional datasets
├── train.py              # Script for training a custom YOLOv11 model (optional)
├── static/
│   └── css/
│       └── style.css     # UI Styling (Neumorphism & Animations)
└── templates/
    └── index.html        # Frontend Dashboard
```

## 📖 Custom Training (Optional)
If you wish to fine-tune a specialized model (e.g., `yolo11n.pt`) on a custom dataset rather than relying on YOLO-World's zero-shot capabilities:
1. See `DATASETS_AND_REFS.md` for links to standard waste datasets.
2. Structure your dataset into `dataset/images/` and `dataset/labels/`.
3. Run `python train.py`. 

## 📜 License
This project is for educational purposes. Feel free to fork, modify, and integrate it into your own environmental and sustainability initiatives!
