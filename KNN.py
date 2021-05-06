#includerea librariilor necesare
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tqdm import tqdm
from skimage.util import img_as_float


#citirea si prelucrarea datelor de train
directory_train = "D:/an II/sem 2/IA/Proiect/train"
path_train = os.path.join(directory_train)

#citirea label-urilor pentru train
train_categ_content = pd.read_csv("D:/an II/sem 2/IA/Proiect/train.txt",sep=",", header = None)
training_data = []

# iteram prin fiecare imagine
i=0
for img in tqdm(os.listdir(path_train)):
    img_array = cv2.imread(os.path.join(path_train,img), cv2.IMREAD_GRAYSCALE)
    img_array = img_as_float(img_array)
    img_array = img_array.flatten()
    training_data.append([img_array,int(train_categ_content[1][i])])
    i = i + 1


#citirea si prelucrarea datelor de validare
directory_validation = "D:/an II/sem 2/IA/Proiect/validation"
path_validation = os.path.join(directory_validation)


validation_categ_content = pd.read_csv("D:/an II/sem 2/IA/Proiect/validation.txt",sep=",", header = None)
validation_data = []
i=0
for img in tqdm(os.listdir(path_validation)):
    img_array = cv2.imread(os.path.join(path_validation, img), cv2.IMREAD_GRAYSCALE)
    img_array = img_as_float(img_array)
    img_array = img_array.flatten()
    validation_data.append([img_array, int(validation_categ_content[1][i])])
    i = i + 1


#citirea si prelucrarea datelor de test
directory_test = "D:/an II/sem 2/IA/Proiect/test"
path_test = os.path.join(directory_test)

test_categ_content = pd.read_csv("D:/an II/sem 2/IA/Proiect/test.txt", sep="\n",header = None)

test_data = []
for img in tqdm(os.listdir(path_test)):
    img_array = cv2.imread(os.path.join(path_test, img), cv2.IMREAD_GRAYSCALE)
    img_array = img_as_float(img_array)
    img_array = img_array.flatten()
    test_data.append([img_array])

#transformarea datelor pentru a putea fi prelucrate
X_train = []
y_train = []
for img,label in training_data:
    print(img,label)
    X_train.append(img)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_validation = []
y_validation = []
for img,label in validation_data:
    X_validation.append(img)
    y_validation.append(label)
X_validation = np.array(X_validation)
y_validation= np.array(y_validation)

X_test = []
for img in test_data:
    X_test.append(img)
X_test = np.array(X_test)

class KNNClassifier:
    #constructorul de initializare
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    #metoda de clasificare care are acces la obiectul curent
    def classify_image(self, test_image, num_neighbors=9, metric='l2'):

        #vom calcula distantele in functie de metrici
        if metric == 'l2':
            distances = np.sum((X_train - test_image) ** 2, axis=-1)
        elif metric == 'l1':
            distances = np.sum(np.abs(X_train - test_image), axis=-1)

        sorted_indexes = np.argsort(distances)

        top_neighbors = self.y_train[sorted_indexes[:num_neighbors]]

        class_counts = np.bincount(top_neighbors)

        return np.argmax(class_counts)

clf = KNNClassifier(X_train, y_train)

predictions = []
for test_image in X_validation:
    pred_label = clf.classify_image(test_image)
    predictions.append(pred_label)

pred_labels = np.array(predictions)
correct_count = np.sum(pred_labels == y_validation)
total_count = len(y_validation)

accuracy = correct_count / total_count

print(f'Accuracy: {accuracy * 100}%')


final_predictions = []
final_img = []
i=0
for final_image in X_test:
    pred_label= clf.classify_image(final_image)
    final_predictions.append(str(pred_label))
    final_img.append(str(test_categ_content[0][i]))
    i = i + 1

d = pd.DataFrame()
d["id"] = final_img
d["label"] = final_predictions
d.to_csv("Sample.csv", index = False)
