"""
Real-time webcam inference with trained OpenCV SVM model.
Press Q to quit.
"""

import os
import json
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "models", "svm.yml")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "models", "label_map.json")

def extract_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    return hog.compute(gray).flatten().astype(np.float32)

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("Missing model. Train first: python src/python/train_classifier.py")
        return

    svm = cv2.ml.SVM_load(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = extract_features(frame).reshape(1, -1)
        _, pred = svm.predict(feat)
        class_idx = str(int(pred[0][0]))
        label = idx_to_label.get(class_idx, "unknown")

        cv2.putText(frame, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 0), 2)
        cv2.imshow("Realtime Inference", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()