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
- Source: Public news datasets used for text classification tasks (from Kaggle)



## ⚙️ How it Works

1. Load and clean the news data using NLP techniques.
2. Convert text into numeric features using **TF-IDF Vectorization**.
3. Train a **PassiveAggressiveClassifier** model.
4. Build a **Streamlit** web app for users to test real/fake headlines interactively.



## 🚀 How to Run

---

## ❗ Important Note for First-Time Users

When you download this project as a ZIP from GitHub and extract it, GitHub may add an extra folder layer like:

 "FakeNewsDetection-main/FakeNewsDetection-main/"

# To avoid file errors like `FileNotFoundError`, make sure you:

1. Open the **inner folder** in your code editor (the one containing `app.py`, `Fake.csv`, etc.)
2. Run the commands **only from that folder** where all files are directly visible.

Or simply:
- Move all files from the inner folder one level up and delete the duplicate folder.



### 1. Install Dependencies

pip install -r requirements.txt

### 2. Run the Streamlit App

streamlit run app.py


### ❓ Handling Misclassification
# Why did the model misclassify a headline?
The model is trained on a limited dataset (mainly political news). It may misclassify headlines from topics it hasn’t seen, like science or space.
📌 To improve accuracy, retraining with a broader and more diverse dataset is recommended.

### 🌐 GitHub Repository
     https://github.com/950280/fake-news-detection.git


👩‍💻 Author
Sindhuja Ganimukkula
AI Internship 2025
Edunet Foundation x Microsoft

