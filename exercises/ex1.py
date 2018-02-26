
def matrix_standardise(matrix):
    """
    features = columns
    samples  = rows
    """

    mean = np.mean(matrix, axis=0)
    std  = np.std(matrix, axis=0)

    normalised_matrix = (matrix - mean) / std

    return normalised_matrix


if __name__ == '__main__':
    m = np.arange(12).reshape(3,4)
    m_std = matrix_standardise(m)
    print("{} vs. {}".format(m, m_std))
