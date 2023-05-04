import pickle

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# print(os.path.abspath(os.getcwd()))
from FileOperations.FileOperations import FileOperations

Categories = ['0', '1']
predicted = list()
actual = list()

model = pickle.load(open('img_model.p', 'rb'))
path = "C:/Users/hrnoz/Desktop/SmallDataSet2"
fileOp = FileOperations(path)
counter = 0
pathOfImages = fileOp.read_images_from_set()
tp = 0
tn = 0
fp = 0
fn = 0
for url in pathOfImages:
    counter += 1
    print(counter)
    img = imread(url)
    img_resize = resize(img, (50, 50, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)
    predicted.append(Categories[model.predict(l)[0]])
    actual.append(int(url[-5]))

    if url[-5] == Categories[model.predict(l)[0]] and int(url[-5]) == 0:
        tn += 1
    if url[-5] != Categories[model.predict(l)[0]] and int(url[-5]) == 1:
        fp += 1
    if url[-5] != Categories[model.predict(l)[0]] and int(url[-5]) == 0:
        fn += 1
    if url[-5] == Categories[model.predict(l)[0]] and int(url[-5]) == 1:
        tp += 1
print(tn, tp, fn, fp)

##CONFUSION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'y_Actual':    actual,
        'y_Predicted': predicted
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


flat_arr = pickle.load(open('flat_arr_28', 'rb'))
tar_arr = pickle.load(open('tar_arr_28', 'rb'))
tar_arr = np.array(tar_arr)
flat_df = np.array(flat_arr)
df1 = pd.DataFrame(flat_df)
df1['Target'] = tar_arr
x1 = df1.iloc[:, :-1]
y1 = df1.iloc[:, -1]
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.1, random_state=77, stratify=y1)
# print(x_test1)
# calculate the fpr and tpr for all thresholds of the classification



import sklearn.metrics as metrics
probs = model.predict_proba(x_test1)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test1, preds)
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