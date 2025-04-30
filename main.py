# Author Classification Project - Full Implementation
# Data Mining Course - Sina

# -----------------------------
# 1. GEREKLİ KÜTÜPHANELERİ YÜKLE
# -----------------------------
import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 2. KLASÖR YAPISINA GÖRE VERİYİ OLUŞTUR
# -----------------------------
def create_dataset(base_dir):
    data = []
    for author_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, author_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            data.append({
                                "author": author_folder,
                                "text": content
                            })
                    except Exception as e:
                        print(f"Hata: {file_path} => {e}")
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("full_dataset.csv", index=False)
    return df

print("🔄 Veri seti oluşturuluyor...")
df = create_dataset("dataset_authorship")
print("✅ Veri seti başarıyla oluşturuldu.")

# -----------------------------
# 3. EĞİTİM VE TEST VERİLERİNİ AYIR
# -----------------------------
X = df['text']
y = df['author']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

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
    "MLP": MLPClassifier(max_iter=1000),
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
    print(f"\n📌 Özellik çıkarım yöntemi: {v_method}")
    vectorizer = get_vectorizer(v_method)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in models.items():
        print(f"  ▶️ Model: {model_name}")
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

        # Sayısal tahminleri geri etikete çevir (yazar isimleri)
        y_pred_names = label_encoder.inverse_transform(y_pred)
        y_test_names = label_encoder.inverse_transform(y_test)

        print("\n🧾 Classification Report:")
        print(f"Model: {model_name}, Özellik: {v_method}")
        print(classification_report(y_test_names, y_pred_names, zero_division=0))

# -----------------------------
# 7. SONUÇLARI GÖSTER
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Model Performans Sonuçları:")
print(results_df.sort_values(by="F1-score", ascending=False))

# CSV’ye kaydet (rapor için)
results_df.to_csv("model_sonuclari.csv", index=False)

# Grafikle görselleştir
sns.set(style="whitegrid")
plt.figure(figsize=(14,6))
sns.barplot(data=results_df, x="Model", y="F1-score", hue="Vectorizer")
plt.xticks(rotation=45)
plt.title("Modellere Göre F1-Skorları")
plt.tight_layout()
plt.show()
