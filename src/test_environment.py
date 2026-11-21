# import numpy
# import pandas
# import matplotlib
# import seaborn
# import sklearn

# print("Everything is working!")


import pandas as pd

data = {
    "ESR1": [8.2, 1.8, 7.4, 2.1],
    "ERBB2": [3.1, 8.7, 2.9, 7.9],
    "MKI67": [2.4, 7.3, 2.8, 9.1],
    "Subtype": ["Luminal A", "HER2", "Luminal A", "Basal"]
}

df = pd.DataFrame(data)

print(df)