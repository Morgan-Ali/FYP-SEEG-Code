import time
import numpy as np
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

num=10
 
labels = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/label.npy')

data = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/psd_feature.npy')

t0 = time.time()

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=20200220)

# Create the decision tree
dt = DecisionTreeClassifier(splitter="random", random_state=20200220, criterion="gini")

# Define RFECV with 100 steps
rfecv = RFECV(estimator=dt, step=100, cv=sss, scoring='accuracy', min_features_to_select=5)

# Fit the RFECV
rfecv.fit(data, labels)
X_selected = rfecv.transform(data)

# Initialize the average accuracy and confusion matrix lists
accuracy_list = []
confusion_matrices = []

# Cross-validation
for train_index, test_index in sss.split(X_selected, labels):

    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Fit the classifier
    dt.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = dt.predict(X_test)

    # Calculate the accuracy and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

    # Calculate the confusion matrix and append to the list
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

mean_cm = np.mean(confusion_matrices, axis=0)

print(f"Mean Accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")

t1 = time.time()
total = t1-t0

print(total)

fig = plt.figure("DT Confusion")
cm_plot = sns.heatmap(mean_cm,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
plt.show()