#/usr/bin/env python3

import numpy as np


def task_a_matrix_standardization(X):
    """
    X: n*d matrix:
    n rows of samples -> one row = 1 sample,
    d dimension columns

    mean -> for every column

    """

    my_mean = np.mean(X, axis=0)
    centered_data = X - my_mean
    normalised_data = centered_data / np.std(centered_data, axis=0)

    return normalised_data


def task_b_eucledian_distance(P, Q):
    print(np.meshgrid(P, Q))
    return np.linalg.norm(np.meshgrid(P, Q), axis=0).ravel()



    # Solution for another problem
    # if P.shape[0] < Q.shape[0]:
    #     num_lines = P.shape[0]
    # else:
    #     num_lines = Q.shape[0]

    # P_cut = P[:num_lines, :]
    # Q_cut = Q[:num_lines, :]

    # return np.linalg.norm(Q_cut - P_cut, axis=1)



# >>> np.stack([ np.repeat([0], 5),  np.arange(5)] ).T
# array([[0, 0],
#        [0, 1],
#        [0, 2],
#        [0, 3],
#        [0, 4]])

# >>> np.repeat(np.arange(5), 2).reshape(-1, 2)
# array([[0, 0],
#        [1, 1],
#        [2, 2],
#        [3, 3],
#        [4, 4]])

# 3D matrix approach (?)

# Complex number approach
# Norm of a complex number = it's length as a vector
# Represent each double pair as the difference of each other?
# 0,2 and -4,7 => 4,5 => sqrt(41)
# 0,2 and -4,7 => -4,-7 => same!
#





# >>> P = [ (0,0), (1,0), (2,0) ]
# >>> Q = [ (1,0), (2,0), (3,0) ]
# >>> np.array(P)
# array([[0, 0],
#        [1, 0],
#        [2, 0]])
# >>> P_np = np.array(P)
# >>> Q_np = np.array(Q)
# >>> Q_np - P_np
# array([[1, 0],
#        [1, 0],
#        [1, 0]])
# >>> import numpy as np
# >>> np.linalg.norm(Q_np - P_np, axis=1)
# array([ 1.,  1.,  1.])

# def task_b_eucledian_distance(P, Q):



if __name__ == '__main__':
    A = np.random.rand(12, 2)
    B = np.random.rand(12, 2)
    C = np.random.rand(14, 2)

    p = [[ 0.5770211 ,  0.43727208], [ 0.2075602 ,  0.96563561],  [ 0.89974397 , 0.71276513] ,  [ 0.45142461 , 0.46265782]]
    q = [[ 0.06996521 , 0.57177429] , [ 0.03982235 ,  0.80206858] ,  [ 0.97983539 ,  0.05024302] , [ 0.19862208 ,  0.31296209] , [ 0.24386778 ,  0.137138  ]]

    print("{}\n {}\n {}".format(A, B, C))


    print(task_a_matrix_standardization(A))
    print(np.std(task_a_matrix_standardization(A), axis=0))
    print(np.mean(task_a_matrix_standardization(A), axis=0))

    print("B : " + str(task_b_eucledian_distance(q, p)))
#    print(task_b_eucledian_distance(A, B))
#    print(task_b_eucledian_distance(A, C))
