# 🏥 AI-Powered Pneumonia Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-green)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

---

## 📌 Project Overview

This project is an AI-powered web application that detects **Pneumonia from Chest X-ray images** using Deep Learning (ResNet18 CNN) and provides intelligent medical explanations through an integrated AI chatbot (LLaMA 3 via Groq API).

The system allows users to:

- Upload chest X-ray images
- Get AI-based pneumonia prediction
- View confidence score
- Generate downloadable PDF medical report
- Chat with AI medical assistant
- View diagnosis history

---

## 🚀 Features

### 🧠 AI Pneumonia Detection
- Built using PyTorch
- Transfer Learning with ResNet18
- Classifies:
  - NORMAL
  - PNEUMONIA
- Displays confidence percentage
- Image preprocessing using Torchvision

---

### 💬 AI Medical Chatbot
- Powered by LLaMA 3 (Groq API)
- Explains diagnosis clearly
- Encourages doctor consultation
- Does NOT prescribe medication

---

### 📄 PDF Report Generation
Generated report includes:
- Uploaded X-ray image
- Diagnosis result
- Confidence score
- Timestamp
- Medical interpretation
- Disclaimer

---

### 📊 Diagnosis History
- Stores previous predictions
- Allows PDF download anytime
- Tracks number of NORMAL vs PNEUMONIA cases

---

## 🛠 Tech Stack

### Backend
- Python
- Flask
- Flask-Login
- Flask-SQLAlchemy
- PyTorch
- Torchvision

### Frontend
- HTML
- Bootstrap
- Chart.js

### AI
- ResNet18 (CNN)
- LLaMA 3 (LLM via Groq API)

### Database
- SQLite

---

## 🧠 Algorithms Used

- Convolutional Neural Networks (CNN)
- Transfer Learning
- CrossEntropyLoss
- Adam Optimizer
- Image Augmentation
- Large Language Model (LLaMA 3)

---

## 📊 Model Details

| Property | Value |
|----------|--------|
| Architecture | ResNet18 |
| Input Size | 224x224 |
| Classes | NORMAL / PNEUMONIA |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Validation Accuracy | ~75–85% |

---

---
## 📁 Project Structure
pythonProject4/
│
├── app.py
├── chatbot.py
├── model.py
├── train_pneumonia.py
├── pneumonia_model.pth
├── database.db
│
├── dataset/
├── static/
│ └── uploads/
│
├── templates/
│ ├── index.html
│ ├── history.html
│ ├── chatbot.html'


<img width="1600" height="846" alt="image" src="https://github.com/user-attachments/assets/e1775ed2-8e8e-4b8a-8134-a7d51d63a0d0" />
<img width="1600" height="830" alt="image" src="https://github.com/user-attachments/assets/759e4067-591b-43f1-97a6-0784b98703d1" />

