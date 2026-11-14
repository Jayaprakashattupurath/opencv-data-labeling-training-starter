/**
 * Node API for lightweight labeling workflow.
 * Stores labels as JSON Lines so training scripts can stream/read incrementally.
 */
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "5mb" }));

const DATA_DIR = path.join(__dirname, "..", "data");
const LABELS_FILE = path.join(DATA_DIR, "labels.jsonl");
fs.mkdirSync(DATA_DIR, { recursive: true });

app.get("/api/health", (_, res) => {
  res.json({ ok: true, service: "cv-label-api" });
});

app.post("/api/label", (req, res) => {
  const { imageId, label, split = "train", quality = "ok", notes = "" } = req.body;

  if (!imageId || !label) {
    return res.status(400).json({ error: "imageId and label are required." });
  }

  const record = {
    imageId,
    label,
    split,      // train | val | test
    quality,    // ok | blurry | occluded
    notes,
    ts: new Date().toISOString()
  };

  fs.appendFileSync(LABELS_FILE, JSON.stringify(record) + "\n", "utf8");
  res.json({ saved: true, record });
});

app.get("/api/guidance", (_, res) => {
  res.json({
    bestPractices: [
      "Keep classes balanced and include hard negatives.",
      "Use train/val/test split by scene/time/device to avoid leakage.",
      "Mark low-quality samples (blurry/occluded) instead of deleting all of them.",
      "Version labels and model files together for reproducibility."
    ]
  });
});

const port = process.env.PORT || 3001;
app.listen(port, () => {
  console.log(`Label API running on http://localhost:${port}`);
});