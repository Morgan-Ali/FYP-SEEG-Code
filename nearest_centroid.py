import time
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt

num = 17

# Load SEEG data labels
labels= np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/label.npy')

data = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/psd_feature.npy')

t0 = time.time()

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)

# KBest selector
kbest = SelectKBest(mutual_info_classif, k=10)

# Generate NCC
ncc = NearestCentroid()

accuracy_scores = []
conf_matrices = []

# Loop through each split of the data
for train_index, test_index in sss.split(data, labels):
    
    # Split data into train and test sets
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Apply KBest on training data
    X_train_selected = kbest.fit_transform(X_train, y_train)
    X_test_selected = kbest.transform(X_test)
    
    ncc.fit(X_train_selected, y_train)
    
    # Predict labels on transformed test data
    y_pred = ncc.predict(X_test_selected)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
print("Accuracy: {:.2f}% (+/- {:.2f}%)".format(np.mean(accuracy_scores)*100, np.std(accuracy_scores)*100))
mean_conf_matrix = np.mean(conf_matrices, axis=0)

t1 = time.time()

total = t1-t0

print(total)

fig = plt.figure("NCC Confusion")
cm_plot = sns.heatmap(mean_conf_matrix,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
#plt.ion()
plt.show()

