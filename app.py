import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/Resume.csv")

df = df[['Resume_str', 'Category']]

df.columns = ['resume_text', 'category']


# ----------------------------
# Clean Text
# ----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


df['cleaned_resume'] = df['resume_text'].apply(clean_text)


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("AI Resume Screening System")

st.write("Paste Job Description Below")


job_description = st.text_area("Job Description")


if st.button("Rank Candidates"):

    scores = []

    for resume in df['cleaned_resume']:

        documents = [job_description, resume]

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        scores.append(similarity)

    df['match_percentage'] = [
        score * 100 for score in scores
    ]

    ranked_df = df.sort_values(
        by='match_percentage',
        ascending=False
    )

    st.subheader("Top Matching Candidates")

    st.dataframe(
        ranked_df[
            ['category', 'match_percentage']
        ].head(10)
    )