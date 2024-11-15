import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
train_sample = np.loadtxt('train1.txt', delimiter=',')
train_label = train_sample[:, 0].astype(int)
train_data = train_sample[:, 1:4]

model = svm.SVC(gamma="scale")
model.fit(train_data, train_label)

test_sample = np.loadtxt('test1.txt', delimiter=',')
test_label = test_sample[:, 0].astype(int)
test_data = test_sample[:, 1:4]

print('Correct Label  :', test_label)
print('Predicted Label:', model.predict(test_data))
print('Test Score: ', model.score(test_data, test_label))