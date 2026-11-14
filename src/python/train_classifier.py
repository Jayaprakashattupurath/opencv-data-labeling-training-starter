"""
Minimal OpenCV training script (HOG + Linear SVM) for image classification.
This demonstrates best-practice flow, not production accuracy.
"""

import os
import json
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")     # expected files: {imageId}.jpg
LABELS_FILE = os.path.join(DATA_DIR, "labels.jsonl")
MODEL_DIR = os.path.join(DATA_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_records():
    if not os.path.exists(LABELS_FILE):
        return []
    records = []
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def extract_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    feat = hog.compute(gray).flatten()
    return feat.astype(np.float32)

def main():
    records = [r for r in load_records() if r.get("split", "train") == "train" and r.get("quality", "ok") == "ok"]
    if not records:
        print("No train records found in labels.jsonl")
        return

    labels = sorted(list({r["label"] for r in records}))
    label_to_idx = {name: i for i, name in enumerate(labels)}

    X, y = [], []
    for r in records:
        image_path = os.path.join(IMAGES_DIR, f"{r['imageId']}.jpg")
        if not os.path.exists(image_path):
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(label_to_idx[r["label"]])

    if len(X) < 10:
        print("Too few valid samples. Collect more labeled data.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    svm.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in label_to_idx.items()}, f, indent=2)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Label map saved: {LABEL_MAP_PATH}")
    print(f"Trained on {len(X)} samples across {len(labels)} classes.")

if __name__ == "__main__":
    main()