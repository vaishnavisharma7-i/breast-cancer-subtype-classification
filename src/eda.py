import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

os.makedirs("results/figures", exist_ok=True)
X = pd.read_csv("data/processed/X_tumour.csv", index_col=0)
y = pd.read_csv("data/processed/y_tumour.csv", index_col=0).iloc[:, 0]
assert (X.index == y.index).all()

order = ["LumA", "LumB", "Her2", "Basal", "Normal"]
col = dict(zip(order, sns.color_palette("Set2", 5)))

# 1. Subtype distribution (class imbalance)
counts = y.value_counts()[order]
plt.figure(figsize=(6, 4))
plt.bar(order, counts, color=[col[o] for o in order])
for i, v in enumerate(counts):
    plt.text(i, v + 3, str(v), ha="center")
plt.ylabel("Samples"); plt.title("PAM50 subtype distribution (522 tumours)")
plt.tight_layout(); plt.savefig("results/figures/subtype_distribution.png", dpi=150); plt.close()

# 2. Expression value distribution
vals = X.values.flatten()
vals = vals[~np.isnan(vals)]
print("Expression min/median/max:", vals.min(), np.median(vals), vals.max())
plt.figure(figsize=(6, 4))
plt.hist(np.random.default_rng(0).choice(vals, 300000), bins=100)
plt.xlabel("Median-centred expression"); plt.ylabel("Count")
plt.title("Expression value distribution")
plt.tight_layout(); plt.savefig("results/figures/expression_distribution.png", dpi=150); plt.close()

# 3. PAM50 gene heatmap, samples sorted by subtype
genes = pd.read_csv("data/processed/pam50_genes_found.csv").iloc[:, 0].tolist()
Z = X[genes].fillna(X[genes].median())
Z = (Z - Z.mean()) / Z.std()
idx = y.map({s: i for i, s in enumerate(order)}).sort_values(kind="stable").index
plt.figure(figsize=(12, 8))
sns.heatmap(Z.loc[idx].T.clip(-3, 3), cmap="vlag", center=0,
            xticklabels=False, yticklabels=True, cbar_kws={"label": "z-score"})
bounds = np.cumsum(y.loc[idx].value_counts(sort=False)
                   .reindex(order).values)
for b in bounds[:-1]:
    plt.axvline(b, color="black", lw=1)
plt.title("PAM50 genes across samples (grouped: " + ", ".join(order) + ")")
plt.tight_layout(); plt.savefig("results/figures/pam50_heatmap.png", dpi=150); plt.close()

# 4. PCA on the 2000 most variable genes (no labels used to fit)
top = X.var().nlargest(2000).index
Xi = SimpleImputer(strategy="median").fit_transform(X[top])
pca = PCA(n_components=2).fit(Xi)
P = pca.transform(Xi)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
plt.figure(figsize=(6, 5))
for s in order:
    m = (y == s).values
    plt.scatter(P[m, 0], P[m, 1], s=12, color=col[s], label=s)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.title("PCA of tumour samples")
plt.tight_layout(); plt.savefig("results/figures/pca_subtypes.png", dpi=150); plt.close()

print("Saved 4 figures to results/figures/")