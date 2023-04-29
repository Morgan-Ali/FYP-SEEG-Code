import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

labels = np.load('/content/label.npy')

data = np.load('/content/psd_feature.npy')

channels = 56

data = data.reshape(1500, channels, 9, -1)

learning_rate = 0.001

# Create an instance of the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Split the training data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=20200220)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=20200220)

num_classes = 5

# Convert the labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Build the DCNN
model = Sequential()
model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(channels, 9)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train_one_hot, epochs=100, batch_size=32)

# Evaluate
score = model.evaluate(X_test, y_test_one_hot, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

fig = plt.figure("DCNN Confusion")
cm_plot = sns.heatmap(cm,annot=True)
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
    #plt.ion()
plt.show()