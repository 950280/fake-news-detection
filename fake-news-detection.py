import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Loading datasets (fake and real)
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv") 

# Adding label column to each
fake['label'] = 'Fake'
real['label'] = 'Real'

# Combining both datasets into one
data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data

# Cleaning the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)              # Remove special characters
    text = text.lower()                          # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


data['title_clean'] = data['title'].apply(clean_text)


# Features and Labels
X = data['title_clean']           # Input title
y = data['label']          # Target labels (Real or Fake)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=2)
X = vectorizer.fit_transform(X)   # Learn and transform

# Split data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Make predictions and check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# User input to predict real or fake news
def predict_news():
    # Load the saved model and vectorizer
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    while True:
        # Get user input
        user_input = input("Enter a news headline to classify as 'real' or 'fake': ")
          
        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye! ")
            break  
        # Clean the input text (same way as training data)
        user_input_clean = clean_text(user_input)
        
        # Convert the user input into the same format as the training data (TF-IDF vectorization)
        user_input_vec = vectorizer.transform([user_input_clean])
    
        # Predict using the trained model
        prediction = model.predict(user_input_vec)
    
        # Output the result
        if prediction == 'Fake':
            print("The news is Fake.")
            print(" Type 'exit' to quit")
        else:
            print("The news is Real.")
            print("Type 'exit' to quit")
            
print("Note: This model is trained on specific datasets of 'real' and 'fake' news.")
print("It may not classify accurately if the input is very different from what it was trained on.\n")
# Call the function to predict user input
predict_news()