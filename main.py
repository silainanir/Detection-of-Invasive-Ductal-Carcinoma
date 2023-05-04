import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from FileOperations.FileOperations import FileOperations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ImageRgbValues.RGB import RGB


def chartForAttributes(total_images, pathList):
    data = pd.DataFrame(index=np.arange(0, total_images), columns=["r", "g", "b", "class"])

    k = 0
    for i in pathList:
        splitList = i.split("/")
        data.iloc[k]["path"] = str(i)
        data.iloc[k]["target"] = int(splitList[6])
        data.iloc[k]["patient_id"] = str(splitList[5])
        # print(data.iloc[k])
        k += 1

    cancer_perc = data.groupby("patient_id").target.value_counts() / data.groupby("patient_id").target.size()
    cancer_perc = cancer_perc.unstack()

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.distplot(data.groupby("patient_id").size(), ax=ax[0], color="Purple", kde=False, bins=30)
    ax[0].set_xlabel("Number of Images")
    ax[0].set_ylabel("Patient Count")
    ax[0].set_title("Numbers of Images for each Patient")

    sns.distplot(cancer_perc.loc[:, 1] * 100, ax=ax[1], color="Red", kde=False, bins=30)
    ax[1].set_title("Ratio of The Images Those show IDC")
    ax[1].set_ylabel("Patient Count")
    ax[1].set_xlabel("Percentage of IDC")

    sns.countplot(data.target, palette="Set3", ax=ax[2])
    ax[2].set_xlabel("Non-IDC (class=0) versus IDC (class=1)")
    ax[2].set_title("Total Number of Images show IDC")
    plt.show()


def process_and_show(rgb, path, patient_folder_names):
    # db = "BreastCancer"
    # dbm = DatabaseManager.DatabaseManager()
    # dbm.create_con(db)
    # table_names = ["Patients", "Pictures0", "Pictures1"]
    r0 = {}
    g0 = {}
    b0 = {}
    r1 = {}
    g1 = {}
    b1 = {}
    for i in range(256):
        r0[i] = g0[i] = b0[i] = r1[i] = g1[i] = b1[i] = 0

    # patient_number = 0
    total_images = 0
    for patient_id in patient_folder_names:
        # if patient_number == 3:
        #     break
        # patient_number += 1

        class0_list, class1_list = rgb.get_image_paths("{}/{}".format(path, patient_id))
        # total_images += len(class1_list) + len(class0_list)
        path_of_patient = "{}/{}".format(path, patient_id)
        class0_rgb = []
        class1_rgb = []

        for j in class0_list:
            absolute_path = "{}/{}/{}".format(path_of_patient, 0, j)
            class0_rgb += rgb.get_rgb_code(absolute_path)
        for j in class1_list:
            absolute_path = "{}/{}/{}".format(path_of_patient, 1, j)
            class1_rgb += rgb.get_rgb_code(absolute_path)

        counter = 0
        for i in class0_rgb:
            r0[i[0]] += 1
            g0[i[1]] += 1
            b0[i[2]] += 1
            counter += 1

        counter = 0
        for i in class1_rgb:
            r1[i[0]] += 1
            g1[i[1]] += 1
            b1[i[2]] += 1
            counter += 1

    keys = list(r0.keys())
    values = list(r0.values())
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='r', ax=ax[0])
    ax[0].set_xlabel("Red Values")
    ax[0].set_ylabel("Red Count")
    ax[0].set_title("Red Values of All Images with Class 0")

    keys = list(g0.keys())
    values = list(g0.values())
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='g', ax=ax[1])
    ax[1].set_xlabel("Green Values")
    ax[1].set_ylabel("Green Count")
    ax[1].set_title("Green Values of All Images with Class 0")

    keys = list(b0.keys())
    values = list(b0.values())
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='b', ax=ax[2])
    ax[2].set_xlabel("Blue Values")
    ax[2].set_ylabel("Blue Count")
    ax[2].set_title("Blue Values of All Images with Class 0")

    plt.show()

    keys = list(r1.keys())
    values = list(r1.values())
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='r', ax=ax[0])
    ax[0].set_xlabel("Red Values")
    ax[0].set_ylabel("Red Count")
    ax[0].set_title("Red Values of All Images with Class 1")

    keys = list(g1.keys())
    values = list(g1.values())
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='g', ax=ax[1])
    ax[1].set_xlabel("Green Values")
    ax[1].set_ylabel("Green Count")
    ax[1].set_title("Green Values of All Images with Class 1")

    keys = list(b1.keys())
    values = list(b1.values())
    sns.histplot(x=keys, weights=values, discrete=False, binwidth=5, color='b', ax=ax[2])
    ax[2].set_xlabel("Blue Values")
    ax[2].set_ylabel("Blue Count")
    ax[2].set_title("Blue Values of All Images with Class 1")

    plt.show()

    data = pd.DataFrame(index=np.arange(0, total_images), columns=["red", "green", "blue"])
    k = 0
    for i in range(len(r0)):
        data.iloc[k]["red"] = list(r0.values())[k] + list(r1.values())[k]
        data.iloc[k]["green"] = list(g0.values())[k] + list(g1.values())[k]
        data.iloc[k]["blue"] = list(b0.values())[k] + list(b1.values())[k]
        # print(data.iloc[k])
        k += 1

    fig = px.scatter_matrix(data, dimensions=["red", "green", "blue"])
    fig.show()

    """
    Extracted means making 1D list out of 2D list. Below class0_rgb and class1_rgb 
    are transformed into 1D list.
    """
    # class0_rgb_extracted = []
    # class1_rgb_extracted = []
    # for i in class0_rgb:
    #     for j in i:
    #         class0_rgb_extracted.append(j)
    # for i in class1_rgb:
    #     for j in i:
    #         class1_rgb_extracted.append(j)

    # dbm.insert_into_columns(table_names[0], ['id', 'totalPictures', 'path'], [patient_id, total_images, path_of_patient])
    # dbm.insert_into_columns(table_names[1], ['r', 'g', 'b'], class0_rgb)
    # dbm.insert_into_columns(table_names[2], ['r', 'g', 'b'], class1_rgb)

    # dbm.cursor.close()


if __name__ == '__main__':
    path = "Dataset"
    """
    RGB -> List Döndürsün
    File -> Listi Bastırsın
    """

    fileOp = FileOperations(path)
    # DataSetten image patlerini okuyorum
    pathOfImages = fileOp.read_images_from_set()
    patient_folder_names = fileOp.read_patient_folders()
    # Image Pathlerini texte yazdırıyorum
    # Parametreler --> (list, fileName, mode/def=append,write\)
    # fileOp.write_2_text(pathOfImages, "pathOfImages.txt", "w")

    # Pathlerin olduğu listi RGB classına veriyorum
    rgb = RGB(pathOfImages)
    # Tüm rgb değerleri list şeklinde dönderiyor.
    """ listRgb-->>> list 
            image_name1,[list of RGB values],image_name2,....."""

    # listRgb = rgb.get_RGB_color_codes()
    process_and_show(rgb, path, patient_folder_names)

    # for i in range(1000):
    #     fileOp.write_2_text(listRgb[i][0:17], "Output.txt", "a")

    # fileOp.write_2_text(listRgb, "rgb.txt")  # -->optional

    # for i in range(10):
    #     print(listRgb[i][0:17])

    """image 50x50'lik list. Her index

      1 2 3 4 5 6 --- 50 (width)
    1 ------------>>>> 
    2
    3
    4
    -
    50 (height)
    """
