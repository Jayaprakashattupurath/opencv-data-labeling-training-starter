# OpenCV Data Labeling + Training Best Practice Starter

A small, practical starter codebase for a **real-time computer vision scenario**:

- Example scenario: **PPE detection pre-workflow** (e.g., helmet vs no-helmet class collection)
- **Node.js API** for label metadata
- **Python + OpenCV** for lightweight training and real-time inference
- **React UI** snippet for quick labeling workflow

---

## Why this structure?

For many real-world CV projects, failure happens in data quality, not model code.  
This starter focuses on **best practices**:

1. Consistent labels
2. Train/val/test split discipline
3. Quality flags (`ok`, `blurry`, `occluded`)
4. Versioning of labels + model artifacts

---

## Project layout

- `src/index.js` → Node API for label records
- `src/python/train_classifier.py` → OpenCV HOG + SVM training example
- `src/python/realtime_infer.py` → webcam inference
- `src/frontend/*` → React labeling helper UI
- `data/` (created at runtime) → labels and trained model files

---

## Quick start

## 1) Backend (Node)

```bash
npm init -y
npm i express cors
node src/index.js
```

API runs at `http://localhost:3001`.

---

## 2) Python (OpenCV)

```bash
python -m venv .venv
# activate venv (Windows: .venv\Scripts\activate, macOS/Linux: source .venv/bin/activate)
pip install opencv-python numpy
python src/python/train_classifier.py
python src/python/realtime_infer.py
```

> Training expects image files in `data/images/{imageId}.jpg` plus label metadata in `data/labels.jsonl`.

---

## 3) Frontend (React snippet)

Use your preferred React toolchain (Vite/CRA).  
Mount `src/frontend/main.jsx` and `src/frontend/App.jsx` in your app.

---

## Real-time use case guidance (practical)

If you are building a real-time safety monitor (factory entry, construction site, lab PPE check):

- Capture examples from **different cameras**, times of day, and distances
- Include hard negatives (similar objects/background)
- Keep at least 10–20% validation data from **different sessions**
- Track annotation mistakes and relabel regularly
- Retrain periodically with new failure cases from production

---

## Data labeling best practices checklist

- [ ] Label definitions are documented (what qualifies as each class)
- [ ] Class balance checked weekly
- [ ] Low-quality frames are tagged, not silently removed
- [ ] No leakage between train and test (same video chunk in both)
- [ ] Model + label snapshot saved together

---

## Notes

- This is intentionally lightweight for learning and prototyping.
- For production, consider:
  - object detection models (YOLO/SSD)
  - MLOps versioning (DVC/MLflow)
  - model drift monitoring and automated re-label pipelines