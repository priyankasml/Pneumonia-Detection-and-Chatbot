from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from datetime import datetime
from io import BytesIO
import os
import random
from fpdf import FPDF
from chatbot import chatbot_response

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

history_data = []
latest_prediction = None


# ---------------- MOCK MODEL ----------------
def mock_predict(path):
    prediction = random.choice(["NORMAL", "PNEUMONIA"])
    confidence = round(random.uniform(85, 99), 2)
    stage = "Healthy" if prediction == "NORMAL" else "Infection Detected"
    comment = (
        "No signs of pneumonia."
        if prediction == "NORMAL"
        else "Pneumonia detected. Please consult a doctor."
    )
    return prediction, confidence, stage, comment


# ---------------- PDF ----------------
def generate_pdf(record):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Pneumonia AI Diagnostic Report", ln=True, align="C")
    pdf.ln(10)

    # Insert X-ray image
    image_path = os.path.join("static/uploads", record["filename"])
    if os.path.exists(image_path):
        pdf.image(image_path, x=30, w=150)
        pdf.ln(85)

    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Patient X-ray File: {record['filename']}", ln=True)
    pdf.cell(0, 10, f"Diagnosis: {record['prediction']}", ln=True)
    pdf.cell(0, 10, f"Confidence Level: {record['confidence']}%", ln=True)
    pdf.cell(0, 10, f"Stage: {record['stage']}", ln=True)
    pdf.cell(0, 10, f"Generated On: {record['timestamp']}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10,
        "Medical Interpretation:\n"
        "This report is generated using AI-based chest X-ray analysis. "
        "If pneumonia is detected, it indicates possible lung infection. "
        "Further clinical evaluation by a healthcare professional is strongly recommended."
    )

    pdf.ln(5)
    pdf.multi_cell(0, 10,
        "Disclaimer:\n"
        "This AI-generated report is for educational purposes only and does not replace professional medical diagnosis."
    )

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    pdf_file = BytesIO(pdf_bytes)
    pdf_file.seek(0)
    return pdf_file

# ---------------- HOME / DASHBOARD ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    global latest_prediction

    uploaded_image = None
    result = confidence = stage = comment = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename:
            uploaded_image = file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_image)
            file.save(path)

            result, confidence, stage, comment = mock_predict(path)

            record = {
                "filename": uploaded_image,
                "prediction": result,
                "confidence": confidence,
                "stage": stage,
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M")
            }

            history_data.append(record)

            latest_prediction = {
                "prediction": result,
                "confidence": confidence,
                "stage": stage,
                "comment": comment
            }

    normal_count = len([r for r in history_data if r["prediction"] == "NORMAL"])
    pneumonia_count = len([r for r in history_data if r["prediction"] == "PNEUMONIA"])

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        stage=stage,
        comment=comment,
        uploaded_image=uploaded_image,
        history=history_data,
        normal_count=normal_count,
        pneumonia_count=pneumonia_count
    )


# ---------------- HISTORY ----------------
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
        pdf_file = generate_pdf(history_data[record_index])
        return send_file(pdf_file, download_name="Report.pdf", as_attachment=True)
    return redirect(url_for("view_history"))


# ---------------- CHATBOT ----------------
@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    reply = chatbot_response(data["message"], latest_prediction)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)