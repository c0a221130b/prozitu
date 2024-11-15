import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)
import matplotlib.pyplot as plt

train = np.loadtxt('mouse_x.csv', delimiter=',')
train_label = train[:, 0].astype(int)
train_data = train[:, 1:65]

test = np.loadtxt('mouse_x_test.csv', delimiter=',')
test_label = test[:, 0].astype(int)
test_data = test[:, 1:65]

model = RandomForestClassifier(random_state=0)
model = model.fit(train_data, train_label)

pred_label = model.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_label, pred_label, pos_label=1)

print('Correct Label  :', test_label)
print('Predicted Label:', pred_label)
print('Test Accuracy Score: ', accuracy_score(pred_label, test_label))
print('Test AUC Score: ', auc(fpr, tpr))

# features = train_X.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(6,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), indices)
plt.show()