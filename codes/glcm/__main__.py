from os import listdir

import numpy as np
import os
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
import cv2
from skimage.measure import label


def feature_extraction_training_datasets():
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'hue', 'value', 'saturaton', 'path', label]
    properties = np.zeros(6)
    final = []
    crop = "potato"
    folders = ["early_blight", "healthy", "late_blight"]
    for folder in folders:
        print(folder)
        labell = folders.index(folder)
        INPUT_SCAN_FOLDER = "../../data/" + crop + "/training/" + folder + "/"

        image_folder_list = os.listdir(INPUT_SCAN_FOLDER)

        for i in range(len(image_folder_list)):

            print(image_folder_list[i])

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
    filepath = crop + "Final.xlsx"
    df.to_excel(filepath)


def feature_extraction_test_datasets():
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    featlist = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'hue', 'value', 'saturaton', 'path',
                'label']
    properties = np.zeros(6)
    final = []
    crop = "potato"
    folders = ["test"]
    for folder in folders:
        labell = folders.index(folder)
        INPUT_SCAN_FOLDER = "../../data/" + crop + "/test/" + folder + "/"

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

            print(image_folder_list[i])

            glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
            for j in range(0, len(proList)):
                properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

            features = np.array(
                [properties[0], properties[1], properties[2], properties[3], properties[4], h_mean, s_mean, v_mean,
                 INPUT_SCAN_FOLDER + image_folder_list[i], labell])
            final.append(features)

    df = pd.DataFrame(final, columns=featlist)
    filepath = crop + "_test_Final.xlsx"
    df.to_excel(filepath)


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


def classify_disease(leaf_name):
    path = leaf_name + 'Final.xlsx'  # give path where extracted features are saved
    path_test = leaf_name + '_test_Final.xlsx'  # give path where the test extracted features are saved

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
    models.append(("KNN", KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree')))

    for name, model in models:
        model.fit(X_train, y_train)
        joblib.dump(model, "../cnn/savedModels/potato" + name + ".pkl")  # Path where the model is going to be saved
        score = model.score(t_test, l_test)
        names.append(name)
        results.append(score * 100)
        saved.append(model)
        print(name, score * 100)

    i = np.argmax(results)

    print("\n\nmax:", names[i], '  ', results[i])

    model = saved[i]

    l_pred = model.predict(t_test)
    prom = np.bincount(l_pred).argmax()  # np.average(l_pred)
    # print('Predicción:', l_pred, '\nReal:', l_test, '\nProm:', int(prom))
    return prom


plant_disease_folder_list = listdir(f"../../data/potato/training/")
feature_extraction_training_datasets()
feature_extraction_test_datasets()
result = classify_disease('potato')
print('Resultado:', plant_disease_folder_list[result])
