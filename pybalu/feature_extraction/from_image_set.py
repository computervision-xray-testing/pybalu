__all__ = ['from_image_set']

from pybalu.base import FeatureExtractor
from sklearn.base import TransformerMixin, clone
from sklearn.pipeline import Pipeline
import numpy as np
import tqdm


def from_image_set(image_set, extractor, show=False):
    '''\
from_image_set(image_set, extractor, show=False)

Returns a 2-dimensional array on which each row represents a different image from the given
image set, and each column a feature extracted by one of the given extractors. 

Parameters
----------
image_set : an ImageSet object
    Represents the set of images over which to extract the selected features. All images
    within the set will have their features extracted.
extractor : an object that implements the sklearn transformer interface
    Can be any of the extractors and image processors defined by PyBalu. Other useful classes inclue
    `sklearn.pipeline.Pipeline` and `sklearn.pipeline.FeatureUnion` to combine multiple extractors.
show : bool, optional
    If set to true, a progress bar that shows the current and estimated completion time will 
    be shown during execution. Default is False.

Returns
-------
features : a 2-dimensional ndarray
    An array with len(`image_set`) rows and a number of columns that depend on the number and
    nature of the given FeatureExtractors
labels : a one dimensional string ndarray
    With the same length as number of columns in `features`, labels each column corresponding
    to the feature it represents.

Examples
--------
Extract the basic intensity features and haralick features for distance 3 for all the images
in a given set

>>> from pybalu.io import ImageSet, print_features
>>> from pybalu.feature_extraction import BasicIntExtractor, HaralickExtractor, from_image_set
>>> from sklearn.pipeline import make_union
>>> 
>>> image_set = ImageSet('./images/', extension='.png')
>>> basic_int, haralick = BasicIntExtractor(), HaralickExtractor(distance=3)
>>> extractors = make_union(basic_int, haralick)
>>> features = from_image_set(image_set, 
...                           extractors,
...                           show=True)
Extracting Features(basic_int): 100%|███████████████████████| 1.68k/1.68k [00:12<00:00, 138.2 imgs/s]
Extracting Features(haralick): 100%|███████████████████████| 1.68k/1.68k [00:13<00:00, 129.9 imgs/s]
>>> labels = np.hstack([basic_int.get_labels(), haralick.get_labels()])
>>> print_features(labels, features[0])
Intensity Mean        :  104.13101
Intensity StdDev      :  22.29537
Intensity Kurtosis    :  4.27058
Intensity Skewness    : -0.89083
Mean Laplacian        : -3.79495
Mean Boundary Gradient: -1.00000
Tx 1 , d 3  (mean)    :  0.27961
Tx 2 , d 3  (mean)    :  0.35839
Tx 3 , d 3  (mean)    :  330.11359
Tx 4 , d 3  (mean)    :  0.35839
Tx 5 , d 3  (mean)    :  0.86984
Tx 6 , d 3  (mean)    :  7.50346
Tx 7 , d 3  (mean)    :  5.94866
Tx 8 , d 3  (mean)    :  1.55481
Tx 9 , d 3  (mean)    :  1.80165
Tx 10, d 3  (mean)    :  0.07122
Tx 11, d 3  (mean)    :  0.64535
Tx 12, d 3  (mean)    : -0.37320
Tx 13, d 3  (mean)    :  0.55530
Tx 14, d 3  (mean)    :  1.44741
Tx 1 , d 3  (range)   :  0.31881
Tx 2 , d 3  (range)   :  0.57074
Tx 3 , d 3  (range)   :  336.75712
Tx 4 , d 3  (range)   :  0.57074
Tx 5 , d 3  (range)   :  0.91066
Tx 6 , d 3  (range)   :  7.54292
Tx 7 , d 3  (range)   :  6.05999
Tx 8 , d 3  (range)   :  1.63802
Tx 9 , d 3  (range)   :  1.97060
Tx 10, d 3  (range)   :  0.08353
Tx 11, d 3  (range)   :  0.79496
Tx 12, d 3  (range)   :  0.50197
Tx 13, d 3  (range)   :  0.66518
Tx 14, d 3  (range)   :  1.48388
    '''
    if not isinstance(extractor, (TransformerMixin, Pipeline)):
        raise TypeError(
            'extractor object must implement sklearn transformer interface'
        )

    old_params = extractor.get_params()
    new_params = dict((k, show) if k.endswith("show") else (k, v)
                      for k, v in old_params.items())
    extractor.set_params(new_params)
    features = extractor.transform(image_set)
    extractor.set_params(old_params)
    return features
