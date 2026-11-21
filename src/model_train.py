import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

X = pd.read_csv("data/processed/X_tumour.csv", index_col=0)
y = pd.read_csv("data/processed/y_tumour.csv", index_col=0).iloc[:, 0]
pam50 = pd.read_csv("data/processed/pam50_genes_found.csv").iloc[:, 0].tolist()
labels = ["LumA", "LumB", "Her2", "Basal", "Normal"]

# Everything below is fitted on training folds only (no leakage)
def make(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-8)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=300)),
        ("clf", model),
    ])

models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "LogReg": LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=0, n_jobs=-1),
    "SVM": SVC(kernel="linear", C=0.1, class_weight="balanced"),
}
experiments = {
    "A_all_genes": X,
    "B_without_PAM50_genes": X.drop(columns=pam50, errors="ignore"),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
rows, pooled = [], {}
for en, Xe in experiments.items():
    for mn, m in models.items():
        acc, bal, f1, yt, yp = [], [], [], [], []
        for tr, te in cv.split(Xe, y):
            pipe = make(clone(m)).fit(Xe.iloc[tr], y.iloc[tr])
            pred = pipe.predict(Xe.iloc[te])
            truth = y.iloc[te]
            acc.append(accuracy_score(truth, pred))
            bal.append(balanced_accuracy_score(truth, pred))
            f1.append(f1_score(truth, pred, average="macro"))
            yt += list(truth); yp += list(pred)
        pooled[(en, mn)] = (yt, yp)
        rows.append(dict(experiment=en, model=mn, n_genes=Xe.shape[1],
                         accuracy=np.mean(acc), accuracy_sd=np.std(acc),
                         balanced_acc=np.mean(bal), macro_f1=np.mean(f1),
                         macro_f1_sd=np.std(f1)))
        print(f"{en:24s} {mn:13s} acc={np.mean(acc):.3f} "
              f"bal_acc={np.mean(bal):.3f} macroF1={np.mean(f1):.3f}")

res = pd.DataFrame(rows).round(3)
res.to_csv("results/metrics/cv_summary.csv", index=False)
print("\n", res.to_string(index=False))

# Model comparison figure
plt.figure(figsize=(8, 4.5))
sns.barplot(data=res, x="model", y="macro_f1", hue="experiment")
plt.ylabel("Macro F1 (5-fold x 3 repeats)"); plt.title("Model comparison")
plt.tight_layout(); plt.savefig("results/figures/model_comparison.png", dpi=150); plt.close()

# Confusion matrix + per-class report for the best real model in each experiment
for en in experiments:
    sub = res[(res.experiment == en) & (res.model != "Dummy")]
    best = sub.sort_values("macro_f1").iloc[-1].model
    yt, yp = pooled[(en, best)]
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True (PAM50)")
    plt.title(f"{en}: {best}")
    plt.tight_layout()
    plt.savefig(f"results/figures/confusion_{en}.png", dpi=150); plt.close()
    rep = classification_report(yt, yp, labels=labels, digits=3)
    open(f"results/metrics/report_{en}.txt", "w").write(f"{en} / {best}\n{rep}")
    print(f"\n=== {en} / {best} ===\n{rep}")