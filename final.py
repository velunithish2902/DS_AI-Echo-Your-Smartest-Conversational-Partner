import os
import io
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import base64


# ===========================================================
# FUNCTION → CONVERT LOCAL IMAGE TO BASE64
# ===========================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


# ===========================================================
# LOAD BACKGROUND IMAGE (LOCAL FILE)
# ===========================================================
background_path = r"C:\Users\katlee\nithi\.venv\futuristic-scene-with-high-tech-robot-used-construction-industry.jpg"
img_base64 = get_base64_image(background_path)


# ===========================================================
# LOAD MODELS
# ===========================================================
ai_echo_model = joblib.load(r"C:\Users\katlee\nithi\.venv\sentiment_analyzer.joblib")
vectorizer = joblib.load(r"C:\Users\katlee\nithi\.venv\vectorizer_balanced (1).joblib")

# Load dataset
df_echo = pd.read_csv(r"C:\Users\katlee\nithi\.venv\processed_cleaned_reviews.csv")


# ===========================================================
# STREAMLIT PAGE CONFIG
# ===========================================================
st.set_page_config(page_title="AI Echo: Sentiment Analysis", layout="wide")


# ===========================================================
# APPLY BACKGROUND + TEXTAREA BLACK THEME
# ===========================================================
st.markdown(
    f"""
    <style>
    /* Background Image */
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(6px);
    }}

    /* Text Color */
    h1, h2, h3, h4, h5, label, p {{
        color: white !important;
        text-shadow: 1px 1px 2px black;
    }}

    /* Textarea (Review Box) → BLACK BOX */
    textarea {{
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #777 !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }}

    .stTextArea textarea {{
        background-color: #000000 !important;
        color: #ffffff !important;
    }}

    /* Input fields */
    input {{
        background-color: rgba(0,0,0,0.3) !important;
        color: white !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ===========================================================
# MAIN UI
# ===========================================================
st.title("📊 AI Echo: Your Smartest Conversational Partner")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 EDA"])


# ===========================================================
# 🔮 PREDICTION PAGE
# ===========================================================
if page == "🔮 Predict":

    st.sidebar.subheader("Sentiment Prediction")

    text_input = st.text_area("Enter your review:")

    if st.button("Predict"):

        if text_input.strip():

            # If model is ML (predict)
            if hasattr(ai_echo_model, "predict"):
                text_vectorized = vectorizer.transform([text_input])
                prediction = ai_echo_model.predict(text_vectorized)[0]

                sentiment_map = {
                    0: "Negative",
                    1: "Neutral",
                    2: "Positive",
                }
                sentiment = sentiment_map.get(prediction, str(prediction))

            else:
                # VADER fallback
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text_input)
                c = scores["compound"]
                if c >= 0.05:
                    sentiment = "Positive"
                elif c <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

            st.subheader(f"Predicted Sentiment: **{sentiment}**")

        else:
            st.warning("Please enter a review first.")


# ===========================================================
# 📊 EDA PAGE
# ===========================================================
elif page == "📊 EDA":
    st.sidebar.subheader("Exploratory Data Analysis (EDA)")
    st.sidebar.write("Explore the dataset and visualize sentiment distribution.")
    st.sidebar.markdown("---")

    st.title("📈 Exploratory Data Analysis (EDA)")
    st.write("This section allows you to explore the dataset and visualize sentiment distribution.")


    # dataloader
    if st.button("Load Data"):
        st.write("Data loaded successfully!")
        st.dataframe(df_echo.head(10))

    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.write("This dataset contains user reviews with their corresponding sentiments and ratings.")
    st.write(f"Total Reviews: {df_echo.shape[0]}")
    st.write(f"Total Columns: {df_echo.shape[1]}")
    st.write(f"Columns: {', '.join(df_echo.columns)}")
        # eda query
    query = st.selectbox("Select a query to visualize:", [
            "What is the overall sentiment of user reviews?",
            "How does sentiment vary by rating?",
            "Which keywords or phrases are most associated with each sentiment class?",
            "What is the distribution of sentiment across different ratings?",
            "How has sentiment changed over time?",
            "Do verified users tend to leave more positive or negative reviews?",
            "Are longer reviews more likely to be negative or positive?",
            "Which locations show the most positive or negative sentiment?",
            "Is there a difference in sentiment across platforms (Web vs Mobile)?",
            "Which ChatGPT versions are associated with higher/lower sentiment?",
            "What are the most common negative feedback themes?"
        ])
    if st.button ("Run Query"):
        if query == "What is the overall sentiment of user reviews?":
            sentiment_counts = df_echo['sentiment'].value_counts()
            st.bar_chart(sentiment_counts, use_container_width=True)
            
        elif query == "How does sentiment vary by rating?":
            rating_sentiment = df_echo.groupby('rating')['sentiment'].value_counts().unstack().fillna(0)
            st.bar_chart(rating_sentiment)

        elif query == "Which keywords or phrases are most associated with each sentiment class?":
            sentiments = df_echo['sentiment'].unique()
            for sentiment in sentiments:
                reviews = df_echo[df_echo['sentiment'] == sentiment]['review'].dropna()
                if not reviews.empty:
                    text = " ".join(reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.subheader(f"Word Cloud for {sentiment} Sentiment")
                    st.pyplot(fig)
                else:
                    st.warning(f"No reviews found for {sentiment} sentiment.")

        elif query == "What is the distribution of sentiment across different ratings?":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df_echo, x='rating', hue='sentiment', ax=ax)
            st.pyplot(fig)

        elif query == "How has sentiment changed over time?":
            if 'date' in df_echo.columns:
                dft = df_echo.dropna(subset=['date']).copy()
                dft['date'] = pd.to_datetime(dft['date'], errors='coerce')
                dft = dft.dropna(subset=['date'])
                dft["month"] = dft['date'].dt.to_period('M').dt.to_timestamp()
                sentiment_over_time = dft.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                fig = px.line(
                    sentiment_over_time,
                    x=sentiment_over_time.index,
                    y=sentiment_over_time.columns,
                    labels={'value': 'Count', 'month': 'Month'},
                    title='Sentiment Over Time'
                )
                fig.update_layout(xaxis_title='Month', yaxis_title='Review Count')
                st.plotly_chart(fig)
            else:
                st.warning("The dataset does not contain a 'date' column for time-based analysis.")

        elif query == "Do verified users tend to leave more positive or negative reviews?":
            if 'verified_purchase' in df_echo.columns:
                verified_sentiment = df_echo.groupby('verified_purchase')['sentiment'].value_counts().unstack().fillna(0)
                verified_sentiment = verified_sentiment.reindex(['Yes', 'No'], axis=0, fill_value=0)
                st.bar_chart(verified_sentiment)
            else:
                st.warning("The dataset does not contain a 'verified_purchase' column for user verification status.")

        elif query == "Are longer reviews more likely to be negative or positive?":
            df_echo['review_length'] = df_echo['review'].apply(lambda x: len(str(x).split()))
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_echo, x='sentiment', y='review_length', ax=ax)
            st.pyplot(fig)

        elif query == "Which locations show the most positive or negative sentiment?":
            if 'location' in df_echo.columns:
                location_sentiment = df_echo.groupby('location')['sentiment'].value_counts().unstack().fillna(0)
                st.bar_chart(location_sentiment)
            else:
                st.warning("The dataset does not contain a 'location' column.")

        elif query == "Is there a difference in sentiment across platforms (Web vs Mobile)?":
            if 'platform_grouped' in df_echo.columns:
                platform_sentiment = df_echo.groupby('platform_grouped')['sentiment'].value_counts().unstack().fillna(0)
                platform_sentiment = platform_sentiment.reindex(['Web', 'Mobile'], axis=0, fill_value=0)
                st.bar_chart(platform_sentiment)
            else:
                st.warning("The dataset does not contain a 'platform_grouped' column.")               

        elif query == "Which ChatGPT versions are associated with higher/lower sentiment?":
            if 'version' in df_echo.columns:
                version_sentiment = df_echo.groupby('version')['sentiment'].value_counts().unstack().fillna(0)
                st.bar_chart(version_sentiment)
            else:
                st.warning("The dataset does not contain a 'version' column.")

        elif query == "What are the most common negative feedback themes?":
            negative_reviews = df_echo[df_echo['sentiment'] == 'Negative']['review'].dropna()
            if not negative_reviews.empty:
                text = " ".join(negative_reviews)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No negative reviews found for analysis.")