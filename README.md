# 🎓 Student Performance Analysis & Prediction Dashboard

## Project Overview
This project is a complete end-to-end data analysis and machine learning solution built on a real student dataset. It explores factors that influence academic performance, trains a **Random Forest classifier** to predict a student's performance level, and delivers everything through a clean, interactive **Dash web dashboard** — all in a single Python file.

---

## 📂 Dataset

| Detail | Info |
|---|---|
| **File** | `student_performance_dataset (1).csv` |
| **Records** | 1000+ students |
| **Source** | [Kaggle – Student Performance Dataset](https://www.kaggle.com/) |

### Columns

| Column | Description |
|---|---|
| `student_id` | Unique identifier |
| `name` | Student name |
| `age` | Age in years |
| `city` | City of study (Delhi, Mumbai, Bangalore…) |
| `course` | Enrolled course (Data Science, AI, ML…) |
| `daily_study_hours` | Average hours studied per day |
| `attendance_percent` | Class attendance percentage |
| `exam_score` | Final exam score (0–100) |
| `enrollment_date` | Date of enrollment |
| `performance_level` | Target label: Poor / Average / Good / Excellent |
| `study_efficiency` | Derived efficiency metric |

---

## 🎯 Objective
1. Understand what drives student academic performance.
2. Identify patterns between study habits, attendance, and scores.
3. Train a machine learning model to **predict** a student's performance level.
4. Serve an interactive dashboard so anyone can explore the data and make predictions.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Dash 2.x | Interactive web dashboard framework |
| Dash Bootstrap Components | Responsive UI styling |
| Plotly 5.x | Interactive charts & visualizations |
| Pandas | Data loading, cleaning, transformation |
| NumPy | Numerical operations |
| scikit-learn | Random Forest, label encoding, metrics |

---

## 📁 Project Structure

```
student_performance/
│
├── student_performance_app.py          ← Main application (backend + frontend)
├── student_performance_dataset (1).csv ← Dataset
├── requirements.txt                    ← Python dependencies
└── README.md                           ← This file
```

---

## ⚙️ Setup & Run Steps

### 1 — Clone / download the project
Place `student_performance_app.py`, `requirements.txt`, and the CSV in the **same folder**.

### 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the application
```bash
python student_performance_app.py
```

### 5 — Open the dashboard
Open your browser and go to **http://127.0.0.1:8050**

---

## 🖥️ Dashboard Features

| Tab | What it shows |
|---|---|
| 📊 **Overview** | Performance distribution pie chart, exam score histogram, attendance boxplot, correlation heatmap |
| 🔍 **Deep Dive** | Study hours vs exam score scatter, course-wise performance bars, city-wise average scores |
| 🤖 **Prediction Model** | Feature importance chart, confusion matrix, classification report table |
| 🔮 **Predict a Student** | Interactive sliders — enter any student's details and get an instant prediction with confidence % |
| 📋 **Dataset** | Full searchable, sortable, filterable data table with colour-coded performance rows |

---

## 📊 Key Findings

1. **Exam score** is the single most important predictor of performance level.
2. Students who study **≥ 5 hours/day** are significantly more likely to achieve "Excellent".
3. **Attendance ≥ 85%** strongly correlates with higher performance.
4. **Machine Learning** and **Data Science** courses have the highest proportion of "Excellent" students.
5. The Random Forest model achieves **~80–85% accuracy** on the held-out test set.
6. `study_efficiency` (a derived metric) shows a **negative correlation** with exam score — high efficiency doesn't always mean high scores without consistent attendance.

---

## 📜 License
This project is for educational purposes. Dataset sourced from Kaggle.
