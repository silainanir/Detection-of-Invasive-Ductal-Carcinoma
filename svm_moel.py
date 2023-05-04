import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

Categories = ['Cars', 'Ice cream cone', 'Cricket ball']
print("Type y to give categories or type n to go with classification of Cars,Ice Cream cone and Cricket ball");

while (True):
    check = input()
    if (check == 'n' or check == 'y'):
        break
    print("Please give a valid input (y/n)")
if (check == 'y'):
    print("Enter How Many types of Images do you want to classify")
    n = int(input())
    Categories = []
    print(f'please enter {n} names')
    for i in range(n):
        name = input()
        Categories.append(name)
    print(
        f"If not drive Please upload all the {n} category images in google collab with the same names as given in categories")

flat_data_arr = []
target_arr = []
# please use datadir='/content' if the files are upload on to google collab
# else mount the drive and give path of the parent-folder containing all category images folders.
datadir = "C:/Users/hrnoz/Desktop/SmallDataSet2"
patients = ["9260", "9261",
            "9262",
            "9265",
            "9266",
            "9267",
            "9290",
            "9291",
            "9319",
            "9320",
            "9321",
            "9322",
            "9323",
            "9323",
            "9325",
            "9344",
            "9345",
            "9346",
            "9347",
            "9381",
            "9382",
            "9383",
            "13025",
            "13106",
            "13400",
            "13401",
            "13402",
            "13403",
            "13404",
            "13458",
            "13459",
            "13460",
            "13461",
            "13462"]

for patient in patients:
    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, patient, i)
        print(patient)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (50, 50, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
model = pickle.load(open('img_model.p', 'rb'))
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
print(df)




x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=77, stratify=y)
print('Splitted Successfully')
probs = model.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)
print('The Model is trained well with the given images')
print(model.best_params_)

y_pred = model.predict(x_test)
print("The predicted Data is :")
print(y_pred)

print("The actual data is:")
print(np.array(y_test))

# classification_report(y_pred,y_test)
print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")
print(confusion_matrix(y_pred, y_test))

pickle.dump(model, open('img_model_28.p', 'wb'))
print("Pickle is dumped successfully")


##CONFUSION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'y_Actual':    y_test,
        'y_Predicted': y_pred
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


model = pickle.load(open('img_model_28.p', 'rb'))

# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
