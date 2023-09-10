def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    # привожу все к np.array
    labels = np.array(labels, dtype="float")
    array = np.array(x, dtype="float")

    # разбиение на кластеры
    a = np.column_stack([array, labels])
    a = a[a[:, array[0].size].argsort()]

    # границы класторов в массиве
    clasters_borders = np.unique(a[:, array[0].size], return_index=True)[1]
    clasters_borders = np.concatenate([clasters_borders,
                                       np.array([x.shape[0]])])

    # длины кластеров
    length = (clasters_borders - np.roll(clasters_borders, 1))[1:]
    # print(length)

    # матрица расстояний
    matrix = sklearn.metrics.pairwise_distances(a[:, : array[0].shape[0]])

    # ответ
    ans = 0
    # номер кластера
    cn = 0
    for i in range(array.shape[0]):
        # выделение строки матрицы расстояний
        element_matrix_line = np.array(np.array_split(matrix[i],
                                       clasters_borders),
                                       dtype=object)[1:-1]
        # определение номера класте для i-ого элемента
        if i in clasters_borders and i != cn:
            cn += 1

        if length[cn] != 1:
            s_i = np.sum(element_matrix_line[cn]) / (length[cn] - 1)

        else:
            continue

        tmp = np.array(np.delete(element_matrix_line, cn, axis=0))
        tmp_l = np.delete(length, cn, axis=0)

        if len(tmp) != 0:
            d_i = np.min(np.array(list(map(np.sum, tmp))) / tmp_l)

        else:
            return 0
            # d_i = 0
        if max(d_i, s_i):
            ans += (d_i - s_i) / max(d_i, s_i)
    sil_score = ans / len(a)

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    
    
    C = np.equal(true_labels[:, None], true_labels[None, :])
    L = np.equal(predicted_labels[:, None], predicted_labels[None, :])
    Correctness = C * L

    precision = np.mean(np.sum(Correctness * C, axis=0)/ np.sum(C, axis=0))
    recall = np.mean(np.sum(Correctness * L, axis=0)/ np.sum(L, axis=0))
    score = 2 * (precision * recall)/((precision + recall))
    
    return score
