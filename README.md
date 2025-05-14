# 📰 Fake News Detection using Machine Learning & NLP

This project is developed as part of the AI Internship by **Edunet Foundation in collaboration with Microsoft**.  
It uses **Natural Language Processing** and **Machine Learning** to detect whether a news headline is real or fake.



## 💡 Project Objective

To build a simple AI-based web application that can classify news headlines as **Real** or **Fake** using a machine learning model.



## 🛠️ Tech Stack

- **Python**
- **Pandas & Numpy**
- **NLTK** for text preprocessing
- **Scikit-learn** for model building
- **Streamlit** for web interface
- **Joblib** for saving/loading models



## 📁 Dataset Used

- `Fake.csv` - contains fake news data
- `True.csv` - contains real news data
- Source: Public news datasets used for text classification tasks



## ⚙️ How it Works

1. Load and clean the news data using NLP techniques.
2. Convert text into numeric features using **TF-IDF**.
3. Train a **PassiveAggressiveClassifier** model.
4. Build a Streamlit web app for users to test real/fake headlines.



## 🚀 How to Run

streamlit run app.py



### 1. Install Dependencies

```bash
pip install -r requirements.txt

