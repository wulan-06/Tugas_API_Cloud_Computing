from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pickle
import numpy as np

# Memuat dataset
dataset = load_dataset("Deysi/spam-detection-dataset")
data = dataset['train']

# Membagi data dan label
X = data['text']
y = data['label']

# Membagi data menjadi latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Melatih model
model.fit(X_train, y_train)

# Simpan model setelah melatihnya
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluasi model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Muat model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App
st.title("Spam Detection Web App")

text = st.text_area("Enter your message:")

if st.button("Predict"):
    # Dapatkan probabilitas prediksi
    probabilities = model.predict_proba([text])
    confidence = np.max(probabilities)  # Probabilitas tertinggi
    predicted_class = model.predict([text])[0]

    # Logika klasifikasi berdasarkan confidence
    if confidence <= 0.99:
        result = "Spam"
    else:
        result = "Not Spam"

    st.write(f"The message is classified as: {result}")
    st.write(f"Confidence: {confidence:.2f}")
