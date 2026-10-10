from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import threading
import time

app = Flask(__name__)

# -------------------------------------------------------------
# ZERO-SHOT LEARNING MODEL (YOLO-World)
# This completely replaces the need for training a dataset!
# It can detect ANY word you type here out of the box.
# -------------------------------------------------------------
print("Loading YOLO-World Zero-Shot Model...")
waste_model = YOLO("yolov8s-world.pt")

# Define exactly what we want to detect (NO training required)
CUSTOM_CLASSES = [
    # Dry Waste
    "plastic bottle", "plastic container", "paper", "cardboard", "pen", 
    "metal can", "glass", "drinking glass", "glass jar", "plastic bag", "plastic wrapper",
    
    # Specific Wet Waste (Organic)
    "apple", "banana", "mosambi", "orange", "tomato", "cucumber",
    
    # Generic Wet Waste
    "vegetable", "fruit", "food waste", "organic waste",
    
    # E-Waste
    "phone", "smartphone", "smartwatch", "watch", "laptop", "battery",
    "earbuds", "airpods", "earphones", "electronic cable", "charger", "charging cable",
    
    # Hazardous / Medical
    "syringe", "medicine blister pack"
]
waste_model.set_classes(CUSTOM_CLASSES)

# Map our custom words to the correct bins
BIN_MAP = {
    # DRY
    "plastic bottle": "DRY ♻",
    "plastic container": "DRY ♻",
    "paper": "DRY ♻",
    "cardboard": "DRY ♻",
    "pen": "DRY ♻",
    "metal can": "DRY ♻",
    "glass": "DRY ♻",
    "drinking glass": "DRY ♻",
    "glass jar": "DRY ♻",
    "plastic bag": "DRY ♻",
    "plastic wrapper": "DRY ♻",
    
    # WET
    "apple": "WET 🟢",
    "banana": "WET 🟢",
    "mosambi": "WET 🟢",
    "orange": "WET 🟢",
    "tomato": "WET 🟢",
    "cucumber": "WET 🟢",
    "vegetable": "WET 🟢",
    "fruit": "WET 🟢",
    "organic waste": "WET 🟢",
    "food waste": "WET 🟢",
    
    # E-WASTE
    "phone": "E-WASTE 🔋",
    "smartphone": "E-WASTE 🔋",
    "smartwatch": "E-WASTE 🔋",
    "watch": "E-WASTE 🔋",
    "laptop": "E-WASTE 🔋",
    "battery": "E-WASTE 🔋",
    "earbuds": "E-WASTE 🔋",
    "airpods": "E-WASTE 🔋",
    "earphones": "E-WASTE 🔋",
    "electronic cable": "E-WASTE 🔋",
    "charger": "E-WASTE 🔋",
    "charging cable": "E-WASTE 🔋",
    
    # HAZARDOUS
    "syringe": "HAZARDOUS ⚠️",
    "medicine blister pack": "HAZARDOUS ⚠️"
}

# Use cv2.CAP_DSHOW (DirectShow) to bypass Windows camera driver bugs!
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Lower resolution slightly for much faster processing
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_frame = None
latest_output_frame = None
detection_state = {
    "object_name": "Scanning...",
    "material": "Waiting...",
    "confidence": 0.0,
    "bin": "Waiting..."
}

def inference_thread():
    global current_frame, latest_output_frame, detection_state
    
    while True:
        if current_frame is None:
            time.sleep(0.05)
            continue
            
        frame_to_process = current_frame.copy()
        output = frame_to_process.copy()
        
        # INCREASED RESOLUTION to 640 so it can see tiny objects like earpods!
        # Lowered confidence to 0.05 to force it to detect them.
        results = waste_model.predict(frame_to_process, conf=0.08, imgsz=640, verbose=False)[0]

        current_material = "Unknown"
        current_confidence = 0.0
        
        best_box = None
        
        # 1. Find the most confident detected waste
        valid_boxes = []
        for b in results.boxes:
            b_list = b.xyxy[0].tolist()
            area = (b_list[2] - b_list[0]) * (b_list[3] - b_list[1])
            total_area = 640 * 480
            
            # Reject massive background objects (larger than 40% of screen)
            if area < (total_area * 0.40):
                valid_boxes.append(b)

        if len(valid_boxes):
            best_box_obj = max(valid_boxes, key=lambda b: float(b.conf[0]))
            current_confidence = float(best_box_obj.conf[0])
            best_cls = int(best_box_obj.cls[0])
            best_box = best_box_obj.xyxy[0].tolist()
            current_material = CUSTOM_CLASSES[best_cls]

        # 2. Draw & Update State
        if current_material != "Unknown":
            # Draw bounding box
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{current_material.title()} {current_confidence*100:.1f}%"
            cv2.putText(output, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Update JSON data for UI
            bin_recommendation = BIN_MAP.get(current_material, "Unknown")
            detection_state["object_name"] = current_material.title()
            detection_state["material"] = current_material.title()
            detection_state["confidence"] = round(current_confidence * 100, 1)
            detection_state["bin"] = bin_recommendation
        else:
            detection_state["object_name"] = "Scanning..."
            detection_state["material"] = "Waiting..."
            detection_state["confidence"] = 0.0
            detection_state["bin"] = "Waiting..."

        latest_output_frame = output
        time.sleep(0.01)

# Start background AI thread
threading.Thread(target=inference_thread, daemon=True).start()

def generate_frames():
    global current_frame, latest_output_frame
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # MIRROR EFFECT: Flip the camera horizontally (right hand is right)
        frame = cv2.flip(frame, 1)
        current_frame = frame
        
        # Display the latest frame processed by AI, or raw if not ready
        frame_to_yield = latest_output_frame if latest_output_frame is not None else frame

        _, buffer = cv2.imencode(".jpg", frame_to_yield)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')
               
        # Cap FPS slightly
        time.sleep(0.03)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def status():
    return jsonify(detection_state)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)