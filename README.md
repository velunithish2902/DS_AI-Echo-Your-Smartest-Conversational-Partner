# DS_AI-Echo-Your-Smartest-Conversational-Partner
# 🤖 DS_AI-Echo — Your Smartest Conversational Partner  
### 📊 Sentiment Analysis & Insights Dashboard (Streamlit + ML + NLP)

AI Echo is an interactive **Sentiment Analysis Dashboard** built using **Streamlit**, **Machine Learning**, and **NLP techniques**.  
It analyzes user reviews, predicts sentiment (Positive, Neutral, Negative), and provides powerful EDA visualizations to understand user feedback deeply.

---

# 🚀 Features

### 🔍 **1. Sentiment Classification**
- Uses ML model (Logistic Regression + TF-IDF)
- Uses VADER/TextBlob hybrid for word-level + sentence-level sentiment
- Supports:
  - **Positive**
  - **Neutral**
  - **Negative**

### 🎨 **2. Modern UI & Dashboard**
- Futuristic AI-themed background  
- Black glass review input box  
- Responsive, clean design  

### 📈 **3. Interactive Visualizations**
Includes multiple insights:

#### **Sentiment Insights**
- Overall sentiment distribution  
- Sentiment by rating  
- Sentiment by platform (Web / Mobile)  
- Sentiment by ChatGPT version  
- Sentiment by user location  
- Sentiment over time (monthly trends)  
- Verified vs non-verified sentiment  

#### **Text Insights**
- Word clouds for each sentiment
- Common negative feedback themes
- Review length distribution by sentiment

---

# 🧠 **Machine Learning Pipeline**

### ✔ Preprocessing:
- Lowercasing  
- Special character removal  
- Stopword removal (negators kept)
- Lemmatization using WordNet  
- POS-aware normalization  
- Missing value handling  
- Platform grouping (Web, Mobile, Other)

### ✔ Sentiment Labeling:
- Based on Rating  
  - `>= 4 → Positive`
  - `3 → Neutral`
  - `<= 2 → Negative`

- Additional VADER-based compound scoring  
- Word-level hybrid sentiment rules  
- Final ensemble sentiment classification  

### ✔ Model Training:
- Balanced dataset (upsampling)
- TF-IDF Vectorizer (1–2 grams, 20k features)
- Logistic Regression classifier
- Stratified train-test split (80/20)
- Saved models using joblib:
  - `sentiment_analyzer.joblib`
  - `vectorizer_balanced.joblib`
  - `text_classifier_balanced.joblib`

---

# 📁 **Dataset Requirements**

Your dataset should contain at least:

| Column | Description |
|--------|-------------|
| `review` | User review text |
| `rating` | Rating 1–5 |
| `date` | Review date |
| `verified_purchase` | Yes/No |
| `location` | User's location |
| `platform` | App Store, Play Store, Web, etc. |
| `version` | ChatGPT version |
| `cleaned_reviews` | Preprocessed cleaned text |
| `sentiment` | Final sentiment label |

---

# 🛠 **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/DS_AI-Echo.git
cd DS_AI-Echo
