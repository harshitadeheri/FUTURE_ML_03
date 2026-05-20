# AI Resume Screening System

An AI-powered Resume Screening and Candidate Ranking System built using Machine Learning and Natural Language Processing (NLP).

This project automatically analyzes resumes, compares them with a given job description, and ranks candidates based on skill and profile similarity.

---

# Project Objective

The goal of this project is to help recruiters and HR teams:

- Screen resumes automatically
- Identify top candidates
- Match resumes with job descriptions
- Reduce manual hiring effort
- Improve recruitment efficiency

---

# Features

- Resume text preprocessing
- NLP-based text cleaning
- TF-IDF vectorization
- Cosine similarity matching
- Candidate ranking system
- Match percentage scoring
- Streamlit web application
- CSV export support
- Visualization of candidate scores

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- Streamlit
- Matplotlib
- Seaborn

---

# Dataset Used

Dataset Source:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Dataset contains:
- Resume text
- Resume categories
- Multiple candidate profiles

---

# Project Structure

```bash
FUTURE_ML_03/
│
├── data/
│   ├── Resume.csv
│   ├── job_description.txt
│   └── resumes/
│
├── notebooks/
│   └── Resume_Screening_System.ipynb
│
├── outputs/
│   └── ranked_candidates.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
