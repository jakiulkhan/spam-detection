# SMS Spam Classification with SVM

Classifies SMS messages as **spam** or **ham** using TF-IDF + Linear SVM.

Trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 messages.

---

## Results

Evaluated on a held-out 20% test set (1,034 messages) after removing 403 duplicates.

| Metric | Score |
|---|---|
| Accuracy | **98.45%** |
| Precision (spam) | **98.32%** |
| Recall (spam) | **89.31%** |
| F1 (spam) | **93.60%** |
| 5-fold CV mean | **97.95%** |


## Dataset
[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — download `spam.csv` and place it here as `SMS.csv`.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook SMS_Spam_SVM.ipynb