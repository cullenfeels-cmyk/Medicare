# Medicare AI: Personalized Healthcare Recommendation System

<div align="center">

  <p><strong>A production-grade AI-powered healthcare analytics and disease diagnosis platform.</strong></p>

  <p>
    <b>Developed By:</b> GULAFSHA <br>
    <b>Program:</b> Zidio Development Internship Project <br>
    <b>Duration:</b> August 2026 – September 2026 (1 Month)
  </p>

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [AI Techniques & Algorithms](#-ai-techniques--algorithms)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Report](#-project-report)

---

## 🚀 About the Project
**Medicare AI** is an advanced artificial intelligence and machine learning web application designed to bridge the gap between predictive clinical analytics and personal health management. The platform processes multi-symptom user profiles alongside vital medical parameters (such as age, gender, blood pressure, and cholesterol levels) to predict potential health conditions, assess patient risk levels, and deliver tailored pharmaceutical, dietary, and lifestyle recommendations.

---

## ✨ Key Features
* **Multi-Symptom Disease Prediction:** Dynamically evaluates symptom combinations mapped against 116 distinct medical conditions using trained machine learning models[cite: 9].
* **Automated Risk Stratification:** Categorizes patient health risk into **Low, Medium, or High** based on clinical parameters and dataset patterns[cite: 2, 9].
* **Exact Medication & Treatment Database:** Provides precise pharmaceutical drug names, dosages, dietary guidelines, and precautions[cite: 7, 9].
* **Secure User Portal:** Implements `Flask-Login` and SQLAlchemy for secure user registration, session tracking, login counts, and personal diagnosis history tracking[cite: 9].
* **Developer Admin Panel:** A dedicated command center tracking platform visitor counts, registered accounts, and user login telemetry[cite: 1, 2, 9].

---

## 🏗️ System Architecture & Workflow
The platform follows an end-to-end data science and full-stack web development pipeline:
1. **Data Ingestion & ETL:** Cleaning and processing historical medical data (`Cleaned_Dataset.csv`) to handle feature vectors.
2. **Feature Engineering & Vitals Integration:** Incorporating patient demographics and vitals into the ML feature matrix[cite: 9].
3. **Model Validation & Scoring:** Supervised classification models outputting confidence percentages and top-5 predicted alternatives[cite: 9].
4. **Full-Stack Deployment:** Integrated via a Flask web server, SQLite database storage (`medicare.db`), and a responsive HTML/CSS frontend with dynamic risk badges[cite: 1, 2, 9].

---

## 🧠 AI Techniques & Algorithms
* **Supervised Classification:** Machine learning models combined with label encoding (`disease_encoder.pkl`) to accurately classify disease vectors.
* **Clinical Safety Overrides:** Dynamic rule-based adjustments that escalate risk ratings to **High** if patients present with elevated blood pressure, high cholesterol, or advanced age[cite: 9].
* **Relational Treatment Lookup:** Structured mapping connecting predicted diagnoses directly to pharmaceutical and lifestyle guidance[cite: 9].

---

## 💻 Technology Stack
* **Language:** Python 3.12
* **Backend:** Flask, Flask-SQLAlchemy, Flask-Login, Flask-CORS[cite: 1]
* **Machine Learning & Data Science:** Scikit-learn, Pandas, NumPy, Joblib[cite: 1]
* **Database:** SQLite (`medicare.db`)
* **Frontend:** HTML5, CSS3 (Inter & Poppins typography, responsive grid layouts, custom UI risk badges)[cite: 1, 2]
* **Version Control:** Git & GitHub[cite: 1]

---

## 📂 Project Structure
```text
medicare/
│
├── models/
│   ├── best_model.pkl
│   ├── disease_encoder.pkl
│   └── feature_columns.pkl
│
├── static/                 # CSS & client assets
├── templates/              # HTML templates (index, auth, dashboard, admin)
├── instance/               # SQLite database storage (medicare.db)
├── Cleaned_Dataset.csv     # Historical clinical dataset
├── app.py                  # Main Flask application server
├── requirements.txt        # Python dependencies
├── Medicare_AI_Zidio_Project_Report.docx  # Official internship report
└── README.md
