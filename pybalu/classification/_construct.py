# import numpy as np

# def construct(*args):
#     train = False
#     test = False
#     n_args = len(args)
#     if n_args == 2:
#         #case 2 -> ds, options = Bcl_svm(Xt, options) # testing only
#         X = np.array([])
#         d = np.array([])
#         Xt = args[0]
#         options = args[1]
#         test = True
#     elif n_args == 3:
#         #case 3 -> options = Bcl_svm(X, d, options) # training only
#         X = args[0]
#         d = args[1].astype(float)
#         if X.shape[0] != d.shape[0]:
#             raise Exception('Length of label vector does not match number of instances.')

#         Xt = np.array([])
#         options = args[2]
#         train = True
#     elif n_args == 4:
#         #case 4 -> ds, options = Bcl_svm(X, d, Xt, options) # training & test
#         X = args[0]
#         d = args[1].astype(float)
#         if X.shape[0] != d.shape[0]:
#             raise Exception('Length of label vector does not match number of instances.')

#         Xt = args[2]
#         options = args[3]
#         test = True
#         train = True
#     else:
#         raise Exception('construct: number of input arguments must be 2, 3 or 4.')

#     return train, test, X, d, Xt, options