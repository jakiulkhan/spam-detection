"""
SMS Spam Classification using SVM
Converted from SMS_Spam_SVM.ipynb
Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
"""

import re
import string
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Setup: NLTK data
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: str = "SMS.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    return df


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, outdir: str = "outputs"):
    import os
    os.makedirs(outdir, exist_ok=True)

    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())
    print(df["label"].value_counts())

    df["char_count"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=df, x="label", ax=ax[0])
    sns.histplot(data=df, x="word_count", hue="label", bins=30, ax=ax[1])
    plt.tight_layout()
    plt.savefig(f"{outdir}/eda_counts.png", dpi=120)
    plt.close(fig)

    spam = " ".join(df[df.label == "spam"].message)
    ham = " ".join(df[df.label == "ham"].message)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(WordCloud(background_color="white").generate(spam))
    ax[0].axis("off")
    ax[0].set_title("Spam")
    ax[1].imshow(WordCloud(background_color="white").generate(ham))
    ax[1].axis("off")
    ax[1].set_title("Ham")
    plt.tight_layout()
    plt.savefig(f"{outdir}/wordclouds.png", dpi=120)
    plt.close(fig)
    print(f"EDA plots saved to ./{outdir}/")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [LEMM.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df["clean_message"] = df["message"].apply(clean_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ---------------------------------------------------------------------------
# 4. Train / tune / evaluate
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])


def train_and_evaluate(df: pd.DataFrame, outdir: str = "outputs"):
    import os
    os.makedirs(outdir, exist_ok=True)

    X = df["clean_message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # Baseline model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print("\n=== Baseline SVM ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))
    print(classification_report(y_test, pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{outdir}/confusion_matrix.png", dpi=120)
    plt.close()

    # Hyperparameter tuning
    param_grid = {"svm__C": [0.1, 1, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("\n=== GridSearch ===")
    print("Best Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    # Cross-validation on best estimator
    scores = cross_val_score(
        grid.best_estimator_, df["clean_message"], df["label"], cv=5
    )
    print("\n=== 5-fold CV ===")
    print(scores)
    print("Mean Accuracy:", scores.mean())

    return grid.best_estimator_


# ---------------------------------------------------------------------------
# 5. Prediction helper
# ---------------------------------------------------------------------------
def make_predictor(model):
    def predict_sms(text: str) -> str:
        pred = model.predict([clean_text(text)])[0]
        return "Spam" if pred == 1 else "Ham"
    return predict_sms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_nltk_data()

    print("Loading data...")
    df_raw = load_data("SMS.csv")

    print("\nRunning EDA...")
    run_eda(df_raw)

    print("\nPreprocessing...")
    df = preprocess(df_raw)
    print(df.head())

    print("\nTraining model...")
    model = train_and_evaluate(df)

    print("\n=== Sanity checks ===")
    predict_sms = make_predictor(model)
    print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
    print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))


if __name__ == "__main__":
    main()