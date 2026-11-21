import pandas as pd

EXP = "data/raw/BRCA.exp.547.med.txt"
LAB = "data/raw/BRCA.547.PAM50.SigClust.Subtypes.txt"

# Look at the raw first lines (delimiter and header)
for f in (EXP, LAB):
    with open(f) as fh:
        print(f)
        for _ in range(3):
            print("   ", fh.readline()[:200].rstrip())

exp = pd.read_csv(EXP, sep="\t", index_col=0)
lab = pd.read_csv(LAB, sep="\t")

print("\nExpression shape (rows x cols):", exp.shape)
print("First 5 row names (genes?):", list(exp.index[:5]))
print("First 5 col names (samples?):", list(exp.columns[:5]))
print("Missing values:", int(exp.isna().sum().sum()))
print("Duplicate gene names:", int(exp.index.duplicated().sum()))

print("\nLabel file shape:", lab.shape)
print(lab.head())
print(lab.columns.tolist())
for c in lab.columns[1:]:
    print("\n", c)
    print(lab[c].value_counts(dropna=False).head(10))

# Sample ID overlap, using the first 12 characters (patient barcode)
e_ids = {c[:12] for c in exp.columns}
l_ids = {str(s)[:12] for s in lab.iloc[:, 0]}
print("\nExpression samples:", len(exp.columns), "| label samples:", len(lab))
print("Matched by 12-char ID:", len(e_ids & l_ids))
print("Sample-type codes in expression:",
      pd.Series([c[13:15] for c in exp.columns]).value_counts().to_dict())