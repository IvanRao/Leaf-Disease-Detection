from os import listdir
import os
from skimage.feature import greycomatrix, greycoprops
import cv2
from skimage.measure import label
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

LEAF = "pepper"
IS_SICK = False

LEAF_FOLDER = "../../data/" + LEAF
RESULT = listdir(f"../../data/" + LEAF + "/training/")


def feature_extraction_training_datasets():
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'hue', 'value', 'saturaton', 'path', label]
    properties = np.zeros(6)
    final = []
    folders = listdir(f"{LEAF_FOLDER}/training")

    for folder in folders:
        labell = folders.index(folder)
        INPUT_SCAN_FOLDER = "../../data/" + LEAF + "/training/" + folder + "/"

        image_folder_list = os.listdir(INPUT_SCAN_FOLDER)

        for i in range(len(image_folder_list)):

            abc = cv2.imread(INPUT_SCAN_FOLDER + image_folder_list[i])

            gray_image = cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(abc, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h_mean = np.mean(h)
            h_mean = np.mean(h_mean)

            s_mean = np.mean(s)
            s_mean = np.mean(s_mean)

            v_mean = np.mean(v)
            v_mean = np.mean(v_mean)

            glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
            for j in range(0, len(proList)):
                properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

            features = np.array(
                [properties[0], properties[1], properties[2], properties[3], properties[4], h_mean, s_mean, v_mean,
                 INPUT_SCAN_FOLDER + image_folder_list[i], labell])
            final.append(features)

    df = pd.DataFrame(final, columns=featlist)
    filepath = LEAF + "Final.xlsx"
    df.to_excel(filepath)


def feature_extraction_test_datasets():
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'hue', 'value', 'saturaton', 'path',
                'label']
    properties = np.zeros(6)
    final = []
    folders = listdir(f"{LEAF_FOLDER}/test")

    for folder in folders:
        labell = folders.index(folder)
        INPUT_SCAN_FOLDER = "../../data/" + LEAF + "/test/" + folder + "/"

        image_folder_list = os.listdir(INPUT_SCAN_FOLDER)

        for i in range(len(image_folder_list)):

            abc = cv2.imread(INPUT_SCAN_FOLDER + image_folder_list[i])

            gray_image = cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(abc, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h_mean = np.mean(h)
            h_mean = np.mean(h_mean)

            s_mean = np.mean(s)
            s_mean = np.mean(s_mean)

            v_mean = np.mean(v)
            v_mean = np.mean(v_mean)

            # print(image_folder_list[i])

            glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
            for j in range(0, len(proList)):
                properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

            features = np.array(
                [properties[0], properties[1], properties[2], properties[3], properties[4], h_mean, s_mean, v_mean,
                 INPUT_SCAN_FOLDER + image_folder_list[i], labell])
            final.append(features)

    df = pd.DataFrame(final, columns=featlist)
    filepath = LEAF + "_test_Final.xlsx"
    df.to_excel(filepath)


def classify_disease():
    path = LEAF + 'Final.xlsx'  # give path where extracted features are saved
    path_test = LEAF + '_test_Final.xlsx'  # give path where the test extracted features are saved

    abc = pd.read_excel(path)
    test = pd.read_excel(path_test)
    X = np.array((abc.to_numpy()))
    T = np.array((test.to_numpy()))

    Y = X[:, -1]  # Stores labels in Y variable
    X = X[:, 1:-1]  # Stores features in X variable
    L = T[:, -1]  # Stores labels in L variable
    T = T[:, 1:-1]  # Stores features in T variable

    y_train = Y.astype('int')
    l_test = L.astype('int')  # labels must be of integer type and not float. So are converted to int

    X_train_path = X[:, -1]
    t_test_path = T[:, -1]

    X_train = X[:, :-1]
    t_test = T[:, :-1]

    print('Test path:\n', t_test_path)

    saved = []
    models = []
    results = []
    names = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5, algorithm='auto')))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('GBC', GradientBoostingClassifier(n_estimators=500)))
    models.append(('SGD', SGDClassifier(max_iter=1000)))
    models.append(('MLP', MLPClassifier(hidden_layer_sizes=2048, alpha=0.001, activation='relu', solver='adam')))

    for name, model in models:
        model.fit(X_train, y_train)
        joblib.dump(model, "../cnn/savedModels/" + LEAF + name + ".pkl")  # Path where the model is going to be saved
        score = model.score(t_test, l_test)
        names.append(name)
        results.append(score * 100)
        saved.append(model)
        print(name, score * 100)

    i = np.argmax(results)

    print("\n\nModelo mas exacto:", names[i], results[i])

    model = saved[i]

    l_pred = model.predict(t_test)
    prom = np.bincount(l_pred).argmax()
    # print('Predicción:', l_pred, '\nReal:', l_test, '\nProm:', int(prom))
    return prom


feature_extraction_training_datasets()
feature_extraction_test_datasets()
result = classify_disease()

if RESULT[result] != 'healthy':
    IS_SICK = True
    print('La planta esta enferma. Tiene:', RESULT[result])
else:
    print('La planta esta sana')

