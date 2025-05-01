# Author Classification Project - Full Implementation
# Data Mining Course - Sina

# -----------------------------
# 1. REQUIRED LIBRARIES
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
# 2. LOAD DATA ACCORDING TO FOLDER STRUCTURE
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
                        print(f"Error: {file_path} => {e}")
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("full_dataset.csv", index=False)
    return df

print("🔄 Creating dataset...")
df = create_dataset("dataset_authorship")
print("✅ Dataset created successfully.")

# -----------------------------
# 3. SPLIT DATA INTO TRAINING AND TEST SETS
# -----------------------------
X = df['text']
y = df['author']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

# -----------------------------
# 4. FEATURE EXTRACTION METHODS
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
        raise ValueError("Unknown feature extraction method")

# -----------------------------
# 5. CLASSIFICATION MODELS
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
# 6. TRAINING, PREDICTION, AND EVALUATION
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
    print(f"\n📌 Feature extraction method: {v_method}")
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

        # Convert numerical predictions back to author names
        y_pred_names = label_encoder.inverse_transform(y_pred)
        y_test_names = label_encoder.inverse_transform(y_test)

        print("\n🧾 Classification Report:")
        print(f"Model: {model_name}, Feature: {v_method}")
        print(classification_report(y_test_names, y_pred_names, zero_division=0))

# -----------------------------
# 7. DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Model Performance Results:")
print(results_df.sort_values(by="F1-score", ascending=False))

# Save to CSV (for reporting)
results_df.to_csv("model_sonuclari.csv", index=False)

# Visualize with bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(14,6))
sns.barplot(data=results_df, x="Model", y="F1-score", hue="Vectorizer")
plt.xticks(rotation=45)
plt.title("F1 Scores by Model")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. BERT INTEGRATION
# -----------------------------
from transformers import BertTokenizer, BertModel
import torch # type: ignore
from tqdm import tqdm # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)

# Function to convert texts into BERT embeddings
def get_bert_embeddings(texts, tokenizer, model, max_length=512):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Vectorizing with BERT"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # [CLS] token representation
            embeddings.append(cls_embedding)
    return np.array(embeddings)

print("\n📦 Starting BERT vectorization...")
X_train_bert = get_bert_embeddings(X_train.tolist(), tokenizer, bert_model)
X_test_bert = get_bert_embeddings(X_test.tolist(), tokenizer, bert_model)
print("✅ BERT vectorization completed.")

# Use only dense models (suitable for BERT output)
bert_models = {
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=500),
    "SVM": SVC()
}

# Train and test models using BERT features
for model_name, model in bert_models.items():
    print(f"\n🧠 [BERT] Model: {model_name}")
    model.fit(X_train_bert, y_train)
    y_pred = model.predict(X_test_bert)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Vectorizer": "BERT",
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    })

    y_pred_names = label_encoder.inverse_transform(y_pred)
    y_test_names = label_encoder.inverse_transform(y_test)

    print("\n🧾 Classification Report:")
    print(f"Model: {model_name}, Feature: BERT")
    print(classification_report(y_test_names, y_pred_names, zero_division=0))

# -----------------------------
# 9. DISPLAY UPDATED RESULTS
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Updated Model Performance Results:")
print(results_df.sort_values(by="F1-score", ascending=False))

# Save to CSV
results_df.to_csv("model_sonuclari.csv", index=False)

# Visualize again
sns.set(style="whitegrid")
plt.figure(figsize=(14,6))
sns.barplot(data=results_df, x="Model", y="F1-score", hue="Vectorizer")
plt.xticks(rotation=45)
plt.title("F1 Scores by Model (TF-IDF & BERT)")
plt.tight_layout()
plt.show()
