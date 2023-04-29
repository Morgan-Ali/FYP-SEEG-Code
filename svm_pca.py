import time
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


kernel = 'sigmoid'
C = 1000

num = 32

y = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/label.npy')

X = np.load('C:/Users/Strangelove/Downloads/share_8_good/gesture/preprocessing/P' + str(num)+'/psd_feature.npy')

t0 = time.time()

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)

pca = PCA(n_components=10)

# Create SVM
svm = SVC(kernel=kernel, C=C)

accuracy_scores = []
confusion_matrices = []

# Cross-validate
for train_index, test_index in sss.split(X, y):
    
    # Split data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Apply PCA on training data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Fit SVM
    svm.fit(X_train_pca, y_train)
    
    # Predict labels
    y_pred = svm.predict(X_test_pca)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
print("Accuracy: {:.2f}% (+/- {:.2f}%)".format(np.mean(accuracy_scores)*100, np.std(accuracy_scores)*100))
mean_conf_matrix = np.mean(confusion_matrices, axis=0)

t1 = time.time()

total = t1-t0

print(total)

fig = plt.figure("SVM Confusion")
cm_plot = sns.heatmap(mean_conf_matrix,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
#plt.ion()
plt.show()

