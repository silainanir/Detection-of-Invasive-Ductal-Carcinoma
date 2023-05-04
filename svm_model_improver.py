import time

import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# print(os.path.abspath(os.getcwd()))
from FileOperations.FileOperations import FileOperations

Categories = ['0', '1']

# model = pickle.load(open('img_model.p', 'rb'))

# url = input('Enter URL of Image')

# plt.imshow(img)
# plt.show()
# path = "C:/Users/hrnoz/Desktop/MidSet"
# fileOp = FileOperations(path)
# # DataSetten image patlerini okuyorum
# pathOfImages = fileOp.read_images_from_set()
# for a in pathOfImages:
#     img = imread(a)
#     img_resize = resize(img, (50, 50, 3))
#     l = [img_resize.flatten()]
#     probability = model.predict_proba(l)
#     for ind, val in enumerate(Categories):
#         print(f'{val} = {probability[0][ind] * 100}%')
#     print(a)
#     print("The predicted image is : " + Categories[model.predict(l)[0]])
#     # print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')

model = pickle.load(open('img_model.p', 'rb'))
path = "C:/Users/hrnoz/Desktop/Data Set"
fileOp = FileOperations(path)
counter = 0
pathOfImages = fileOp.read_images_from_set()
for url in pathOfImages:
    counter += 1
    img = imread(url)
    # plt.imshow(img)
    # plt.show()
    img_resize = resize(img, (50, 50, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)
    # for ind, val in enumerate(Categories):
    #     print(f'{val} = {probability[0][ind] * 100}%')
    # print("The predicted image is : " + Categories[model.predict(l)[0]])
    # print("Actual label is: ", url[-5])
    # print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')
    if url[-5] == Categories[model.predict(l)[0]]:
        b = "y"
    else:
        b = "n"
    # print(b)
    # time.sleep(0.1)

    while True:
        # b = input()
        if b == "y" or b == "n":
            break
        # print("please enter either y or n")

    if (b == 'n'):
        print(counter)
        # print("What is the image?")
        # for i in range(len(Categories)):
        #     print(f"Enter {i} for {Categories[i]}")
        k = int(url[-5])
        while k < 0 or k >= len(Categories):
            print(f"Please enter a valid number between 0-{len(Categories) - 1}")
            # k = int(input())
        print("Please wait for a while for the model to learn from this image :)")
        # flat_arr = model.flat_data_arr.copy()
        # AttributeError: 'GridSearchCV' object has no attribute 'flat_data_arr'
        flat_arr = pickle.load(open('flat_arr', 'rb'))
        tar_arr = pickle.load(open('tar_arr', 'rb'))
        tar_arr.append(k)
        flat_arr.extend(l)
        ##
        pickle.dump(flat_arr, open('flat_arr', 'wb'))
        pickle.dump(tar_arr, open('tar_arr', 'wb'))

        tar_arr = np.array(tar_arr)
        flat_df = np.array(flat_arr)
        df1 = pd.DataFrame(flat_df)
        df1['Target'] = tar_arr
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
        svc = svm.SVC(probability=True)
        model1 = GridSearchCV(svc, param_grid)
        x1 = df1.iloc[:, :-1]
        y1 = df1.iloc[:, -1]
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.1, random_state=77, stratify=y1)
        d = {}
        for i in model.best_params_:
            d[i] = [model.best_params_[i]]
        model1 = GridSearchCV(svc, d)
        model1.fit(x_train1, y_train1)
        y_pred1 = model.predict(x_test1)
        print(f"The model is now {accuracy_score(y_pred1, y_test1) * 100}% accurate")
        pickle.dump(model1, open('img_model.p', 'wb'))
        model = pickle.load(open('img_model.p', 'rb'))
    # print("Thank you for your feedback")
