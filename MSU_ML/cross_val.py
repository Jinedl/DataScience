import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    arr = list(range(num_objects))
    step = num_objects // num_folds
    ans = []
    for i in range(0, num_objects-(num_objects % num_folds + step), step):
        ans.append((
                    arr[:i] + arr[(i+step):], arr[i:(i+step)]
                    ))
    ans.append((
                arr[:-(num_objects % num_folds + step)], arr[-(num_objects % num_folds + step):]
                ))
    return ans


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    ans = {}

    for scl in parameters['normalizers']:
        for n in parameters['n_neighbors']:
            for m in parameters['metrics']:
                for w in parameters['weights']:

                    clf = knn_class(n_neighbors=n, metric=m, weights=w)

                    sum = 0
                    count = 0

                    for i in folds:

                        x_train = X[i[0]].copy()
                        y_train = y[i[0]].copy()
                        x_test = X[i[1]].copy()
                        y_true = y[i[1]].copy()

                        if scl[1] != 'None':
                            scl[0].fit(x_train)
                            x_train = scl[0].transform(x_train)
                            x_test = scl[0].transform(x_test)

                        clf.fit(x_train, y_train)
                        y_predict = clf.predict(x_test)

                        sum += score_function(y_true, y_predict)
                        count += 1

                    ans[(scl[1], n, m, w)] = sum / count

    return ans
