# 📌 Machine Learning Model: Spam Detection using Scikit-learn

# STEP 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 2: Load the dataset
# NOTE: Download 'spam.csv' from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# STEP 3: Preprocess labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# STEP 4: Visualize label distribution
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()

# STEP 5: Split data
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# STEP 7: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# STEP 8: Predict and evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 9: Test on custom messages
def predict_message(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Examples
print("\n📨 Test Messages:")
print("1:", predict_message("Congratulations! You've won a $1000 gift card!"))
print("2:", predict_message("Hi, are we still meeting for lunch today?"))
print("3:", predict_message("FREE entry in 2 a weekly competition to win FA Cup tickets"))
