# Author Classification Project - Full Implementation
# Data Mining Course - Sina

# -----------------------------
# 1. GEREKLİ KÜTÜPHANELERİ YÜKLE
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# 2. VERİYİ YÜKLE VE ÖN İŞLE
# -----------------------------
data = pd.read_csv("your_dataset.csv")  # CSV dosyasını bu klasöre koymalısın
data.dropna(inplace=True)
data = data.reset_index(drop=True)

# -----------------------------
# 3. EĞİTİM VE TEST VERİLERİNİ AYIR
# -----------------------------
X = data['text']
y = data['author']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# -----------------------------
# 4. ÖZELLİK ÇIKARIM METOTLARI
# -----------------------------
def get_vectorizer(method):
    if method == "tfidf_word":
        return TfidfVectorizer(analyzer='word')
    elif method == "tfidf_word_2gram":
        return TfidfVectorizer(analyzer='word', ngram_range=(2,2))
    elif method == "tfidf_word_3gram":
        return TfidfVectorizer(analyzer='word', ngram_range=(3,3))
    elif method == "tfidf_char_2gram":
        return TfidfVectorizer(analyzer='char', ngram_range=(2,2))
    elif method == "tfidf_char_3gram":
        return TfidfVectorizer(analyzer='char', ngram_range=(3,3))
    else:
        raise ValueError("Bilinmeyen özellik çıkarım metodu")

# -----------------------------
# 5. SINIFLANDIRMA MODELLERİ
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(),
    "NaiveBayes": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=300),
    "DecisionTree": DecisionTreeClassifier()
}

# -----------------------------
# 6. EĞİTİM, TAHMİN VE DEĞERLENDİRME
# -----------------------------
vector_methods = [
    "tfidf_word",
    "tfidf_word_2gram",
    "tfidf_word_3gram",
    "tfidf_char_2gram",
    "tfidf_char_3gram"
]

results = []

for v_method in vector_methods:
    print(f"\nÖzellik çıkarım yöntemi: {v_method}")
    vectorizer = get_vectorizer(v_method)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in models.items():
        print(f" Model: {model_name}")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            "Vectorizer": v_method,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })

# -----------------------------
# 7. SONUÇLARI GÖSTER
# -----------------------------
results_df = pd.DataFrame(results)
print("\nModel Performans Sonuçları:")
print(results_df.sort_values(by="F1-score", ascending=False))

# Grafikle görselleştir (opsiyonel)
sns.set(style="whitegrid")
plt.figure(figsize=(14,6))
sns.barplot(data=results_df, x="Model", y="F1-score", hue="Vectorizer")
plt.xticks(rotation=45)
plt.title("Modellere Göre F1-Skorları")
plt.tight_layout()
plt.show()
