import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from FileOperations.FileOperations import FileOperations

K_NUMBER = 10
PATH = "Dataset"
N_JOBS = -1


def extract_features(i_path, m_model):  # convert from 'PIL.Image.Image' to numpy array
    img = load_img(i_path, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)  # reshaping image for model
    imgx = preprocess_input(reshaped_img)
    features = m_model.predict(imgx)

    return features


def get_results(k=K_NUMBER):

    general_percentage_clusters = [[] for _ in range(k)]

    for cluster_label in range(k):
        print(cluster_label)
        n_healthy = 0
        n_diseased = 0

        for f_name in groups[cluster_label]:
            if f_name.endswith('1.png'):
                n_diseased += 1
            else:
                n_healthy += 1

        healthy_percentage = n_healthy / len(groups[cluster_label])  # for cluster label #
        disease_percentage = n_diseased / len(groups[cluster_label])  # for cluster label #

        general_percentage = (n_diseased + n_healthy) / (n_class_d + n_class_h)
        general_percentage_clusters[cluster_label].append(general_percentage)
        general_percentage_clusters[cluster_label].append([healthy_percentage, disease_percentage])
        general_percentage_clusters[cluster_label].append(n_healthy)
        general_percentage_clusters[cluster_label].append(n_diseased)
        general_percentage_clusters[cluster_label].append(get_cluster_entropy(n_healthy, n_diseased))

    return general_percentage_clusters


def get_charts(k=K_NUMBER):

    general_sizes_pie = []
    general_sizes_str=[]
    entropy_values = []
    for _clusters in general_percentage_groups:
        _general_percentage = round(_clusters[0]*100, 5)
        general_sizes_pie.append(_general_percentage)
        h_perc = round(_clusters[1][0]*100, 3)
        d_perc = round(_clusters[1][1]*100, 3)
        h_count = _clusters[2]
        d_count = _clusters[3]
        p_entropy = (h_count + d_count)/n_class_d
        v_entropy = (p_entropy * _clusters[4])
        entropy_values.append(v_entropy)
        my_str = ' Healthy:' + str(h_perc) + '%' + "({}),".format(h_count) +\
                 ' Diseased:' + str(d_perc) + "%" + "({})".format(d_count)
        general_sizes_str.append(my_str)

    general_entropy = sum(entropy_values)

    plt.title('K = ' + str(k)+', General Entropy: ' + str(round(general_entropy, 5)))
    patches, texts, juck = plt.pie(general_sizes_pie, startangle=90, radius=1, shadow=True, autopct='%10.1f%%')
    plt.legend(patches, general_sizes_str, loc="lower left", fontsize='large')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def get_cluster_entropy(n_healthy, n_diseased):
    if n_healthy == 0:
        counts = np.asarray([n_diseased])
    elif n_diseased == 0:
        counts = np.asarray([n_healthy])
    else:
        counts = np.asarray([n_healthy, n_diseased])

    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


fileOp = FileOperations(PATH)
pathOfImages = fileOp.read_images_from_set()

print("Loading model...")
model = VGG16()  # loading the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

print("Getting features data...")

features_data = {}
n_class_d = 0  # class 1
n_class_h = 0  # class 0
for img_path in pathOfImages:
    my_features = extract_features(img_path, model)
    features_data[img_path] = my_features

    # setup labels
    if img_path.endswith('1.png'):
        n_class_d += 1
    else:
        n_class_h += 1


a_features = np.array(list(features_data.values()))
a_features = a_features.reshape(-1, 4096)

# PCA

pca = PCA(random_state=69)
pca.fit(a_features)
X = pca.transform(a_features)

k_means = KMeans(n_clusters=K_NUMBER, n_jobs=N_JOBS, random_state=69)
k_means.fit(X)

groups = {}
for f_path, cluster in zip(pathOfImages, k_means.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(f_path)
    else:
        groups[cluster].append(f_path)


general_percentage_groups = get_results()

get_charts()

