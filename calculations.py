from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def la(df, algo):

    x = df.drop(df.columns[0], axis=1)  # features
    y = df[df.columns[0]]  # label
    result = []  # store the results

    enc = OneHotEncoder()  # one-hot encode the features
    x = enc.fit_transform(x).toarray()
    sc = StandardScaler()  # scale the features
    x = sc.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # split the data

    if algo == 'knn':
        knn = KNeighborsClassifier()  # instantiate the model

        param_grid = {  # define the hyperparameters
            'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        grid_knn = GridSearchCV(knn, param_grid, cv=5)  # instantiate the grid search
        grid_knn.fit(x_train, y_train)  # fit the model
        y_pred = grid_knn.predict(x_test)  # make predictions

        result.append(accuracy_score(y_test, y_pred))
        result.append(precision_score(y_test, y_pred, average='macro'))
        result.append(recall_score(y_test, y_pred, average='macro'))
        result.append(f1_score(y_test, y_pred, average='macro'))

        return result

    if algo == 'logreg':
        logreg = LogisticRegression()  # instantiate the model

        param_grid = {  # define the hyperparameters
            'C': [0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }

        grid_logreg = GridSearchCV(logreg, param_grid, cv=5)  # instantiate the grid search
        grid_logreg.fit(x_train, y_train)  # fit the model
        y_pred = grid_logreg.predict(x_test)  # make predictions

        result.append(accuracy_score(y_test, y_pred))
        result.append(precision_score(y_test, y_pred, average='macro'))
        result.append(recall_score(y_test, y_pred, average='macro'))
        result.append(f1_score(y_test, y_pred, average='macro'))

        return result

    if algo == 'svm':
        svm = SVC()  # instantiate the model

        param_grid = {  # define the hyperparameters
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        }

        grid_svm = GridSearchCV(svm, param_grid, cv=5)  # instantiate the grid search
        grid_svm.fit(x_train, y_train)  # fit the model
        y_pred = grid_svm.predict(x_test)  # make predictions

        result.append(accuracy_score(y_test, y_pred))
        result.append(precision_score(y_test, y_pred, average='macro'))
        result.append(recall_score(y_test, y_pred, average='macro'))
        result.append(f1_score(y_test, y_pred, average='macro'))

        return result
