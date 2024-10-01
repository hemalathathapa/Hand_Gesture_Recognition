import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
import flask

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global cap  # Declare cap as a global variable

    ret, frame = cap.read()
    roi = frame[100:400, 320:620]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    result = np.argmax(model.predict(roi.reshape(1, 28, 28, 1)), axis=-1)
    letter = getLetter(result)
    return jsonify({'result': letter})


train = pd.read_csv(r"C:\\Users\\gkmah\\Documents\\train.csv")
test = pd.read_csv(r"C:\\Users\\gkmah\\OneDrive\\Desktop\\Documents\\test.csv")
train.head()

labels = train['label'].values
unique_val = np.array(labels)
np.unique(unique_val)

plt.figure(figsize=(18, 8))
sns.countplot(x=labels)

train.drop('label', axis=1, inplace=True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

index = 2
print(labels[index])
plt.imshow(images[index].reshape(28, 28))

import cv2
import numpy as np

for i in range(0, 10):
    rand = np.random.randint(0, len(images))
    input_im = images[rand]
    sample = input_im.reshape(28, 28).astype(np.uint8)
    sample = cv2.resize(sample, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("sample image", sample)
    cv2.waitKey(0)

cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

batch_size = 128
num_classes = 24
epochs = 10

x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

plt.imshow(x_train[0].reshape(28, 28))

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
model.save("sign_mnist_cnn_50_Epochs.h5")
print("Model Saved")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

test_labels = test['label']
test.drop('label', axis=1, inplace=True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape

y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())


def getLetter(result):
    classLabels = {0: "A",
                   1: "B",
                   2: "C",
                   3: "D",
                   4: "E",
                   5: "F",
                   6: "G",
                   7: "H",
                   8: "I",
                   9: "J",
                   10: "K",
                   11: "L",
                   12: "M",
                   13: "N",
                   14: "O",
                   15: "P",
                   16: "Q",
                   17: "R",
                   18: "S",
                   19: "T",
                   20: "U",
                   21: "V",
                   22: "W",
                   23: "X",
                   24: "Y"}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"


import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    result = np.argmax(model.predict(roi, 1, verbose=0), axis=-1)
    letter = getLetter(result)
    print("Predicted Letter:", letter)  # Print the predicted letter
    cv2.putText(copy, letter, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    roi = roi.reshape(1, 28, 28, 1)
    result = np.argmax(model.predict(roi, 1, verbose=0), axis=-1)
    cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global cap  # Declare cap as a global variable

    ret, frame = cap.read()
    roi = frame[100:400, 320:620]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    result = np.argmax(model.predict(roi.reshape(1, 28, 28, 1)), axis=-1)
    letter = getLetter(result)
    return jsonify({'result': letter})

  





 
if __name__ =='__main__':
    app.run(debug=True)
