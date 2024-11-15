import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_table('table3.tsv', encoding='UTF-8', header=None)

df.columns=['V1', 'V2']
print(df)
print(df['V1'].value_counts())

label = df['V1'].apply(lambda s: 1 if s == 'spam' else 0)
data = pd.DataFrame(df['V2'])

vectorizer = CountVectorizer()
vectorizer.fit(data['V2'])

print('Vocabulary size: {}'.format(len(vectorizer.vocabulary_)))
print('Vocabulary contents: {}'.format(vectorizer.vocabulary_))

bow = vectorizer.transform(data['V2'])
print('bow:\n{}'.format(repr(bow)))