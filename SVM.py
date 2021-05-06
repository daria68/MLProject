#includerea librariilor necesare
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import seaborn as sn
from tqdm import tqdm
from skimage.util import img_as_float
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVC


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
    test_data.append(img_array)


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
print("Valid")
print(X_validation)


X_test = []
for img in test_data:
    X_test.append(img)
X_test = np.array(X_test)
print("Test")
print(type(X_test))
print(X_test)


clf = SVC(C=1, kernel = 'linear')
clf.fit(X_train,y_train)
pred_labels = clf.predict(X_validation)

accuracy = accuracy_score(y_validation,pred_labels)

print("Accuracy:", accuracy)

cm = confusion_matrix(pred_labels,y_validation)
d_cm = pd.DataFrame(cm)
plt.figure(figsize= (7,5))
ax = sn.heatmap(d_cm, annot = True, center = 1500)
ax.set_ylim([0,2])
plt.savefig("confusionmatrix.png")


pred_test = clf.predict(X_test)

final_img = []
final_pred = []
i=0
for final_image in X_test:
    final_img.append(str(test_categ_content[0][i]))
    final_pred.append((str(pred_test[i])))
    i = i + 1

d=pd.DataFrame()
d["id"]=final_img
d["label"]=final_pred
d.to_csv("SVM_1_linear.csv", index=False)
