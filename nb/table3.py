import pandas as pd

df = pd.read_table('table3.tsv', encoding='UTF-8', header=None)

df.columns=['V1', 'V2']
print(df)
print(df['V1'].value_counts())