import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

X = pd.read_csv("data/processed/X_tumour.csv", index_col=0)
y = pd.read_csv("data/processed/y_tumour.csv", index_col=0).iloc[:, 0]
pam50 = set(pd.read_csv("data/processed/pam50_genes_found.csv").iloc[:, 0])
experiments = {"A_all_genes": X,
               "B_without_PAM50_genes": X.drop(columns=list(pam50), errors="ignore")}

def prep():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(1e-8)),
                     ("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=300))])

def selected_names(p, cols):
    return cols[p.named_steps["var"].get_support()][p.named_steps["select"].get_support()]

for en, Xe in experiments.items():
    # NOTE: fitted on all data for interpretation only, not for performance
    pre = prep().fit(Xe, y)
    names = selected_names(pre, Xe.columns)
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    clf.fit(pre.transform(Xe), y)
    coef = pd.DataFrame(clf.coef_.T, index=names, columns=clf.classes_)
    print(f"\n=== {en} ===")
    print("Selected genes that are PAM50 genes:", sum(g in pam50 for g in names), "of 300")

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    tops = []
    for ax, c in zip(axes, clf.classes_):
        top = coef[c].nlargest(10)
        ax.barh(top.index[::-1], top.values[::-1],
                color=["tab:red" if g in pam50 else "tab:gray" for g in top.index[::-1]])
        ax.set_title(c)
        tops.append(pd.DataFrame({"subtype": c, "gene": top.index, "coef": top.values}))
        print(c, ":", ", ".join(top.index[:8]))
    fig.suptitle(f"Top genes per subtype ({en}); red = PAM50 gene")
    plt.tight_layout()
    plt.savefig(f"results/figures/top_genes_{en}.png", dpi=150); plt.close()
    pd.concat(tops).to_csv(f"results/metrics/top_genes_{en}.csv", index=False)

    # Stability: how often is each gene selected across 5 training folds?
    counts = pd.Series(0, index=Xe.columns)
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, y):
        p = prep().fit(Xe.iloc[tr], y.iloc[tr])
        counts[selected_names(p, Xe.columns)] += 1
    stable = counts[counts == 5].index
    print("Genes selected in all 5 folds:", len(stable))
    pd.Series(stable).to_csv(f"results/metrics/stable_genes_{en}.csv", index=False)