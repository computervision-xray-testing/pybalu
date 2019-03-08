__all__ = ['pca']
import numpy as np
from scipy.linalg import eigh

def pca(features, *, n_components=0, energy=0):
    '''\
    pca(features, n_components=0, energy=0)
    
    Principal component analysis

    Parameters
    ----------
    features: numerical 2 dimensional ndarray
        Corresponds to the sample features over which to perform PCA.
    n_components: positive integer, optional
        Number of selected components. If both `n_components` and `energy` are not given,
        `n_components` is set to the original number of features.
    energy: float in range [0, 1], optional
        If `n_components` is not set and `energy` is greater than 0, `n_components` is set
        to the lowest possible number that fulfills the condition
            sum(stdev[:n_components])) / sum(stdev) >= energy.
        where `stdev` corresponds to the standard deviation of each principal component.

    Returns
    -------
    p_components: float ndarray 
        The `n_components` principal components of `features`, sorted according to decreasing
        `stdev`.
    stdev: float ndarray
        The standard deviation of each principal component. One dimensional array 
        of size `n_components`.
    transform: float ndarray
        The transformation matrix for obtaining the principal components.
    feat_mean: float ndarray
        The mean of each feature in `features`.
    new_features: float ndarray
        Of the same shape as `features`. Corresponds to the new features calculated
        from the selected principal components. It corresponds to performing the 
        following operation: `p_components` @ `selected_transform`.T + `feat_mean`

    Examples
    --------
    Reduce a grayscale image to 30 principal components and display together with original
    
    >>> from pybalu.io import imread
    >>> from pybalu.feature_transformation import pca
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> img = imread('cameraman.png', flatten=True)
    >>> _, _, _, _, new_img = pca(img, n_components=30)
    >>> plt.subplot(121); plt.imshow(img, cmap='gray'); plt.subplot(122); plt.imshow(new_image, cmap='gray')
    >>> plt.show()


    Reduce the number of features

    (TODO)

    '''
    N = features.shape[1]

    if not isinstance(n_components, int):
        raise ValueError(f'`n_components` must be a positive int, not {n_components}')
    if n_components <= 0:
        if energy == 0:
            raise ValueError('Either `n_components` or `energy` must be set')
        if not 0 < energy < 1:
            raise ValueError(f'`energy` must be a value between 0 and 1, not {energy}')
        # energy is a valid value
        n_components = N
    else:
        # n_components is a valid value
        energy = 0
    
    feat_mean = features.mean(0)
    X0 = features - feat_mean
    Cx = np.cov(X0, rowvar=False)

    if energy > 0:
        # calculate ALL eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(Cx) 
        # reverse them into descending value 
        _lambda = eig_vals[::-1]
        transform = eig_vecs[:,::-1]

        # calculate the number of selected components
        energy_eig = np.tril(np.ones((N, N))) @ _lambda / _lambda.sum()
        n_components = np.where(energy_eig > energy)[0][0] + 1
        transform = transform[:, :n_components]
        _lambda = _lambda[:n_components]

    else:
        # calculate only the wanted number of eigenvectors
        eig_vals, eig_vecs = eigh(Cx, eigvals=(N-n_components, N-1))
        # reverse them into descending value 
        _lambda = eig_vals[::-1]
        transform = eig_vecs[:,::-1]

    p_components = X0 @ transform
    new_features = p_components @ transform.T + feat_mean

    return p_components, _lambda, transform, feat_mean, new_features