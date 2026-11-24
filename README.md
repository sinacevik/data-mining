# Turkish Authorship Attribution

A text-classification experiment that predicts the author of a Turkish news column from its writing style. The repository compares TF-IDF representations and several classical machine-learning models, with an optional BERT-embedding experiment.

## Problem and data

- **Task:** multi-class authorship attribution
- **Dataset:** 1,200 Turkish articles stored as plain-text files
- **Classes:** 30 authors, with 40 documents per author
- **Split:** 80% training and 20% test data using `random_state=42`
- **Evaluation:** accuracy, weighted precision, weighted recall, and weighted F1

The program builds the labelled dataset directly from the folder names under `dataset_authorship/`.

## Experiment pipeline

1. Read each author's `.txt` files and create a labelled `pandas` DataFrame.
2. Encode author labels and create a fixed train/test split.
3. Compare five TF-IDF representations:
   - word unigrams
   - word bigrams
   - word trigrams
   - character bigrams
   - character trigrams
4. Train and evaluate Random Forest, SVM, XGBoost, Multinomial Naive Bayes, MLP, and Decision Tree classifiers.
5. Optionally generate `bert-base-uncased` embeddings and evaluate dense classifiers on those embeddings.

## Committed results

The strongest result in the committed `model_sonuclari.csv` is the MLP classifier with character-bigram TF-IDF features.

| Representation | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| Character 2-gram TF-IDF | MLP | 0.9500 | 0.9502 |
| Character 3-gram TF-IDF | MLP | 0.9292 | 0.9298 |
| Character 3-gram TF-IDF | SVM | 0.8833 | 0.8849 |
| Character 2-gram TF-IDF | Random Forest | 0.8875 | 0.8825 |

These figures come from one fixed hold-out split. They are useful for comparing the implementations in this repository, but they should not be treated as cross-validated or production benchmarks.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost transformers torch tqdm
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The BERT stage downloads `bert-base-uncased` on first use and can take considerably longer than the TF-IDF experiments. The script writes:

- `full_dataset.csv` - generated labelled dataset
- `model_sonuclari.csv` - model comparison metrics
- interactive charts comparing weighted F1 scores

## Repository layout

```text
.
├── dataset_authorship/   # One folder per author, one text file per article
├── main.py               # Dataset creation, training, evaluation, and plots
└── model_sonuclari.csv   # Committed TF-IDF experiment results
```

## Limitations

- Model-level random seeds are not fixed, so repeated runs may differ.
- The current evaluation uses a single hold-out split rather than cross-validation.
- The BERT stage uses the English-focused `bert-base-uncased` model; a Turkish or multilingual encoder would be a stronger follow-up experiment.
- The repository does not currently document the original dataset source or redistribution licence; confirm those terms before reusing the corpus outside this project.

## Possible next steps

- add stratified cross-validation and deterministic seeds
- compare Turkish and multilingual transformer encoders
- add a reproducible dependency lock file
- save confusion matrices and per-author error analysis
