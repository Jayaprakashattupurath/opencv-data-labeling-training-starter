import React, { useRef, useState } from "react";

const API = "http://localhost:3001";

export default function App() {
  const videoRef = useRef(null);
  const [label, setLabel] = useState("helmet");
  const [message, setMessage] = useState("");

  const startCam = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    setMessage("Camera started. Capture and label samples.");
  };

  const saveLabelOnly = async () => {
    // In production, you'd upload actual image bytes too.
    const imageId = `img_${Date.now()}`;
    const res = await fetch(`${API}/api/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId, label, split: "train", quality: "ok" })
    });
    const data = await res.json();
    setMessage(`Saved label for ${data.record?.imageId || "unknown"}`);
  };

  return (
    <div style={{ fontFamily: "Arial", maxWidth: 700, margin: "20px auto" }}>
      <h2>CV Labeling Helper (React)</h2>
      <p>Use consistent labels and capture varied angles/backgrounds.</p>

      <video ref={videoRef} autoPlay playsInline width="480" style={{ border: "1px solid #ddd" }} />
      <div style={{ marginTop: 12 }}>
        <button onClick={startCam}>Start Camera</button>
      </div>

      <div style={{ marginTop: 12 }}>
        <label>Label:&nbsp;</label>
        <input value={label} onChange={(e) => setLabel(e.target.value)} />
        <button onClick={saveLabelOnly} style={{ marginLeft: 8 }}>Save Label Record</button>
      </div>

      <p style={{ color: "#333" }}>{message}</p>
    </div>
  );
}