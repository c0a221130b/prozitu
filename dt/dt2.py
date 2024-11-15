import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)

train = np.loadtxt('mouse_x.csv', delimiter=',')
train_label = train[:, 0].astype(int)
train_data = train[:, 1:65]

test = np.loadtxt('mouse_x_test.csv', delimiter=',')
test_label = test[:, 0].astype(int)
test_data = test[:, 1:65]

model = DecisionTreeClassifier(random_state=0)
model = model.fit(train_data, train_label)

pred_label = model.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_label, pred_label, pos_label=1)

print('Correct Label  :', test_label)
print('Predicted Label:', pred_label)
print('Test Accuracy Score: ', accuracy_score(pred_label, test_label))
print('Test AUC Score: ', auc(fpr, tpr))