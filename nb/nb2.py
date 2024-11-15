import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

df = pd.read_table('SMSSpamCollection', encoding='UTF-8', header=None)

df.columns=['V1', 'V2']
print(df)
print(df['V1'].value_counts())

label = df['V1'].apply(lambda s: 1 if s == 'spam' else 0)
data = pd.DataFrame(df['V2'])

vectorizer = CountVectorizer(min_df=3, stop_words='english')
vectorizer.fit(data['V2'])

print('Vocabulary size: {}'.format(len(vectorizer.vocabulary_)))
print('Vocabulary contents: {}'.format(vectorizer.vocabulary_))

bow = vectorizer.transform(data['V2'])

train_data, test_data, train_label, test_label = train_test_split(data, label, train_size=0.8)
vectorizer.fit(train_data['V2'])

bow_train = vectorizer.transform(train_data['V2'])
bow_test = vectorizer.transform(test_data['V2'])

print('bow:\n{}'.format(repr(bow_train)))
print('bow:\n{}'.format(repr(bow_test)))

model = BernoulliNB()
model.fit(bow_train, train_label)

print('Stop words: {}'.format(vectorizer.get_stop_words()))

print('Train accuracy: {:.3f}'.format(model.score(bow_train, train_label)))
print('Test accuracy: {:.3f}'.format(model.score(bow_test, test_label)))
