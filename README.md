📧 Spam Message Classifier using Machine Learning

This project is a Python-based spam message classifier that uses Natural Language Processing (NLP) techniques and machine learning (ML) algorithms to detect whether a given SMS/text message is **spam** or **ham (not spam)**.

---

🚀 Features

- Text preprocessing using NLTK (tokenization, stopword removal)
- Vectorization with TF-IDF
- Classification using Multinomial Naive Bayes
- Accuracy, precision, recall, and F1-score evaluation
- Interactive user input for live prediction

---

🧠 How It Works

1. Load and clean the SMS Spam Collection dataset
2. Preprocess messages (lowercasing, tokenizing, stopwords removal)
3. Train/test split using scikit-learn
4. Train a Multinomial Naive Bayes classifier
5. Predict on test set and calculate performance metrics
6. Take live user input to classify custom messages
