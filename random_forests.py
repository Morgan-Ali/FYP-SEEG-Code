import time
import numpy as np
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt


num=32
 
labels = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/label.npy')

data = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/psd_feature.npy')

t0 = time.time()

# SSS into 5 splits, fit all splits to cross-validate
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=20200220)

# Define the RF
rf = RandomForestClassifier(n_estimators=10, random_state=20200220)

# Feature selection with cross-validation, eliminate 100 features per iteration
rfecv = RFECV(estimator=rf, step=100, cv=sss, scoring='accuracy', min_features_to_select=5)

# Fit the RFECV object to the data
rfecv.fit(data, labels)
data_selected = rfecv.transform(data)

# Initialize the accuracy and confusion matrix lists that will be averaged
accuracy_list = []
confusion_matrices = []

# Cross-validation
for train_index, test_index in sss.split(data_selected, labels):

    # Split into train / test data using the SSS as a guide
    X_train, X_test = data_selected[train_index], data_selected[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Fit the classifier to the training data
    rf.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)
mean_cm = np.mean(confusion_matrices, axis=0)

print(f"Mean Accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")

t1 = time.time()
total = t1-t0

print(total)

fig = plt.figure("RF Confusion")
cm_plot = sns.heatmap(mean_cm,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
plt.show()