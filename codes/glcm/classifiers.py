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

path = 'potatoFinal.xlsx'  # give path where extracted features are saved
path_test = 'potato_test_Final.xlsx'  # give path where the test extracted features are saved

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

# y_pred = model.predict(X_test)
# print(y_pred, '\n', y_test)

l_pred = model.predict(t_test)
prom = np.average(l_pred)
print('Predicción:', l_pred, '\nReal:', l_test, '\nProm:', int(prom))

# for file, pred, actual in zip(X_test_path, y_pred, y_test):
# for file, pred, actual in zip(t_test_path, l_pred, l_test):
# print(file)
# img = cv2.imread(file[:-3])
# img = cv2.imread(file, 0)

# cv2.putText(img, folders[pred], (x, y), font, 3, (0, 255, 0), 2)
# cv2.putText(frame, 'actual: ' + sFolder, (0, y + h+20), font, 1, (0, 255, 0))

# cv2.putText(img, folders[actual], (x, y + 30), font, 3, (0, 255, 255), 2)
# cv2.imshow(file, img)
# cv2.waitKey(0)
# cv2.waitKey(1)
# print(file)
