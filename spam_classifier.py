import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')

# Text preprocessing
def preprocess(text):
    text = text.lower()
    tokens = text.split()  
    tokens = [word.strip(string.punctuation) for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
   
    try:
        df = pd.read_csv("spam.csv", encoding='latin-1')
        df = df.iloc[:, :2]  
        df.columns = ['label', 'message']
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

   
    df['cleaned'] = df['message'].astype(str).apply(preprocess)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

   
    print("\n🔍 Spam Detection - Type a message below:")
    while True:
        user_input = input("📩 Your message (or type 'exit'): ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        cleaned_input = preprocess(user_input)
        vect_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vect_input)[0]
        label = 'SPAM 🚫' if prediction == 1 else 'HAM ✅'
        print(f"🔎 Prediction: {label}\n")

if __name__ == "__main__":
    main()
