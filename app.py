from flask import Flask, render_template, request, redirect, url_for, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from datetime import datetime
from io import BytesIO
from fpdf import FPDF  # For PDF generation

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = ["NORMAL", "PNEUMONIA"]

# ---------------- MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- MEMORY HISTORY ----------------
history_data = []

# ---------------- PREDICTION ----------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], round(conf.item() * 100, 2)

def stage_comment(result):
    if result == "NORMAL":
        return "Healthy", "No pneumonia detected."
    else:
        return "Infection Detected", "Pneumonia detected. Please consult a doctor."

# ---------------- PDF GENERATION ----------------
def generate_pdf(record):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Pneumonia Diagnosis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Filename: {record['filename']}", ln=True)
    pdf.cell(0, 10, f"Disease: {record['prediction']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {record['confidence']}%", ln=True)
    pdf.cell(0, 10, f"Stage: {record['stage']}", ln=True)
    pdf.cell(0, 10, f"Timestamp: {record['timestamp']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Comments: {record.get('comment', 'N/A')}")

    # Correct way to generate PDF in memory
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # returns bytes
    pdf_file = BytesIO(pdf_bytes)
    pdf_file.seek(0)
    return pdf_file

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = stage = comment = image = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return redirect(url_for("index"))

        image = file.filename
        path = os.path.join(UPLOAD_FOLDER, image)
        file.save(path)

        result, confidence = predict_image(path)
        stage, comment = stage_comment(result)

        history_data.append({
            "filename": image,
            "prediction": result,
            "confidence": confidence,
            "stage": stage,
            "comment": comment,
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M")
        })

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        stage=stage,
        comment=comment,
        uploaded_image=image,
        history=history_data
    )

@app.route("/history")
def view_history():
    return render_template("history.html", history=history_data)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    history_data.clear()
    return redirect(url_for("view_history"))

@app.route("/download_pdf/<int:record_index>")
def download_pdf(record_index):
    if 0 <= record_index < len(history_data):
        record = history_data[record_index]
        pdf_file = generate_pdf(record)
        return send_file(
            pdf_file,
            download_name=f"Report_{record['filename'].split('.')[0]}.pdf",
            as_attachment=True
        )
    return redirect(url_for("view_history"))

if __name__ == "__main__":
    app.run(debug=True)
