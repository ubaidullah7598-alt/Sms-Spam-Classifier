

SMS Spam Classifier

This project is a simple SMS Spam Classifier that can tell if a text message is spam or ham (not spam) using Naive Bayes.

How It Works

The text messages are cleaned and converted into numbers using a vectorizer.

The Naive Bayes algorithm learns from labeled messages (spam or ham).

It can predict new messages as spam or not.

Features

Detects spam messages.

Easy to use and understand.

Can be improved or added to apps.

How to Run

Clone the project:

git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier

Install dependencies:

pip install -r requirements.txt

Run the script:

python sms_spam_classifier.py

Test with your own message:

message = "You won a free prize!"
prediction = model.predict([message])
print(prediction)  # spam
Dataset

Uses the SMS Spam Collection dataset.

Messages are labeled as spam or ham.

Future Improvements

Build a web or mobile app for real-time SMS filtering.

Try deep learning models for better accuracy.
