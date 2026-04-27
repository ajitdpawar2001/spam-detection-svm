import pandas as pd
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Save model (optional)
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

# ---------------- STREAMLIT ---------------- #

st.title("📩 Spam Detection App")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    data = vectorizer.transform([user_input])
    result = model.predict(data)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")

# Show metrics
st.subheader("📊 Model Performance")
st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")
st.text("Classification Report:\n" + cr)