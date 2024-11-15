import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

samples = np.loadtxt('mouse_x.csv', delimiter=',')
samples_label = samples[:, 0].astype(int)
samples_data = samples[:, 1:65]

model = svm.SVC(gamma="scale")
model.fit(samples_data, samples_label)

tests = np.loadtxt('mouse_x_test.csv', delimiter=',')
tests_label = tests[:, 0].astype(int)
tests_data = tests[:, 1:65]

print('Correct Label  :', tests_label)
print('Predicted Label:', model.predict(tests_data))
print('Test Score: ', model.score(tests_data, tests_label))