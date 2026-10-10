# Real-time Waste Segregation Datasets & References

This document provides links and instructions for finding suitable datasets and open-source references to train your waste segregation model.

## 1. Datasets for Training

Since you are classifying waste into **Dry (Plastic, Paper, Cardboard, Metal, Glass)** and **Wet (Organic, Biodegradable)**, you need datasets that support these classes.

### Kaggle Datasets
1. **Garbage Classification (12 classes) by ColinBorg**
   - Contains: paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, etc.
   - **Link:** [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
2. **Waste Classification Data by TechSash**
   - Contains: Organic and Recyclable images (Good for direct Wet/Dry classification).
   - **Link:** [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
3. **TrashNet Dataset**
   - Contains: glass, paper, cardboard, plastic, metal, trash.
   - **Link:** [TrashNet](https://github.com/garythung/trashnet)

### Roboflow Universe (Highly Recommended for YOLO)
Roboflow provides pre-annotated datasets specifically for YOLO formats.
- **Search Query:** `Waste Segregation YOLO` or `Garbage Classification YOLO`
- **Link:** [Roboflow Universe - Waste Segregation](https://universe.roboflow.com/search?q=waste)
- *Why Roboflow?* You can download the dataset directly in `YOLOv11` format, eliminating the need to manually convert XML/JSON bounding boxes into YOLO txt format.

### Special E-Waste & Earpods Datasets (Requested)
If you wish to train a dedicated model (instead of relying on YOLO-World zero-shot capabilities) for tiny electronic items like Earpods, Earphones, or general E-Waste, here are the best pre-labeled YOLO datasets:
1. **Earphone Detection Dataset (by Meta)** 
   - **Link:** [Earphone Detection - Roboflow](https://universe.roboflow.com/meta-eazrn/earphone-mzmrf)
2. **Earbuds Detection (by Krishna)**
   - **Link:** [Earbuds Detection - Roboflow](https://universe.roboflow.com/krishna-yrqus/earbuds-dectedtion)
3. **Only-Headphones/Earphones**
   - **Link:** [Headphones Dataset - Roboflow](https://universe.roboflow.com/ans401/only-headphones-earphones)

---

## 2. Open-Source Github Repositories (References)
These repositories offer robust software-only implementations of waste classification that you can reference for UI/UX or model architecture:

1. **[TrashNet Implementation in PyTorch](https://github.com/garythung/trashnet)**
   - A classic baseline for waste classification.
2. **[YOLOv8-Garbage-Classification](https://github.com/ultralytics/ultralytics)**
   - You can refer to Ultralytics' own hubs for various custom trained models. You can often find pre-trained weights on their platforms.

---

## 3. How to Train the Model

We have included a `train.py` file which automates the YOLOv11 training process.

### Step 1: Prepare the Dataset
1. Download a YOLO-formatted dataset from Roboflow or Kaggle.
2. Extract it into the `dataset/` folder.
3. Your folder structure should look like this:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── dataset.yaml
   ```

### Step 2: Edit `dataset.yaml`
Ensure `dataset.yaml` lists the correct paths and classes:
```yaml
path: dataset
train: images/train
val: images/val

names:
  0: plastic
  1: paper
  2: cardboard
  3: metal
  4: glass
  5: biodegradable
```

### Step 3: Run the Training Script
Run the provided training script:
```bash
python train.py
```
*Note: It is highly recommended to run this on Google Colab, Kaggle Notebooks, or a machine with a dedicated NVIDIA GPU to speed up training.*

### Step 4: Update the App
Once training is complete, your best model will be saved to `runs/waste_detection/weights/best.pt`.
Replace the old `model/best.pt` with your newly trained `best.pt`.
