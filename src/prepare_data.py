import pandas as pd

EXP = "data/raw/BRCA.exp.547.med.txt"
LAB = "data/raw/BRCA.547.PAM50.SigClust.Subtypes.txt"

exp = pd.read_csv(EXP, sep="\t", index_col=0)
lab = pd.read_csv(LAB, sep="\t").set_index("Sample")

print("Duplicate IDs (exp, lab):",
      exp.columns.duplicated().sum(), lab.index.duplicated().sum())

# Match by full barcode, never by row order
common = exp.columns.intersection(lab.index)
print("Full-ID matches:", len(common))
lab = lab.loc[common]
X = exp[common].T            # samples x genes

print("\nSample type vs PAM50 subtype:")
print(pd.crosstab(lab["Type"], lab["PAM50"]))

# Main analysis: primary tumours only
keep = lab["Type"] == "tumor"
X_t, y_t = X[keep], lab.loc[keep, "PAM50"]
print("\nPrimary tumours:", X_t.shape)
print(y_t.value_counts())
print("Repeated patients among tumours:",
      X_t.index.str[:12].duplicated().sum())

miss = X_t.isna().mean()
print("\nGenes with any missing:", int((miss > 0).sum()),
      "| max missing fraction:", round(miss.max(), 3),
      "| genes >20% missing:", int((miss > 0.2).sum()))

# PAM50 gene overlap (includes common alias names)
pam50 = """ACTR3B ANLN BAG1 BCL2 BIRC5 BLVRA CCNB1 CCNE1 CDC20 CDC6 CDCA1 CDH3
CENPF CEP55 CXXC5 EGFR ERBB2 ESR1 EXO1 FGFR4 FOXA1 FOXC1 GPR160 GRB7 KIF2C
KNTC2 KRT14 KRT17 KRT5 MAPT MDM2 MELK MIA MKI67 MLPH MMP11 MYBL2 MYC NAT1
ORC6L PGR PHGDH PTTG1 RRM2 SFRP1 SLC39A6 TMEM45B TYMS UBE2C UBE2T
NUF2 NDC80 ORC6""".split()
found = [g for g in pam50 if g in X_t.columns]
print("\nPAM50 genes found in matrix:", len(found))
print("Not found:", [g for g in pam50 if g not in X_t.columns])

X_t.to_csv("data/processed/X_tumour.csv")
y_t.to_csv("data/processed/y_tumour.csv")
pd.Series(found).to_csv("data/processed/pam50_genes_found.csv", index=False)
print("\nSaved files to data/processed/")