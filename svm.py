import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import time


# Define hyperparameters
kernel = 'linear'
C = 1

num = 2
 
labels = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/label.npy')

data = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/psd_feature.npy')

t0 = time.time()

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=20200220)

clf = SVC(kernel=kernel, C=C)

# Define the RFECV
rfecv = RFECV(estimator=clf, step=100, cv=sss, scoring='accuracy', min_features_to_select=5)
rfecv.fit(data, labels)

# Extract the selected features
X_selected = rfecv.transform(data)

# Initialize the accuracy and confusion matrix lists
accuracy_list = []
confusion_matrices = []

# Perform the cross-validation
for train_index, test_index in sss.split(X_selected, labels):

    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Fit the classifier 
    clf.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = clf.predict(X_test)
   
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

fig = plt.figure("SVM Confusion")
cm_plot = sns.heatmap(mean_cm,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
plt.show()