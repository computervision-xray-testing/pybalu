__all__ = ["structure"]

import numpy as np


def structure(classifier, *, train_data=None, test_data=None, train_classes=None, initialize=True,
              fit_method='fit', predict_method='predict', classifier_opts=None):
    '''\
    structure(classifier, train_data=None, test_data=None, train_classes=None, initialize=True,
              fit_method='fit', predict_method='predict, classifier_opts=None)

    Trains and/or classifies a set of data using the given classifier. Performs either or both actions
    depending on the given keyword arguments.

    Parameters 
    ----------
    classifier: Class or instance of a Class that defines train and predict methods
        The classifier to train or predict with. It can be a reference to the class or an instance of a class
        that defines two methods:
            - The `fit` method should accept a `train_data` and `train_classes` as input and should train the 
              classifier.
            - The `predict` method should accept `test_data` as input and return a numerical array-like.
        If an instance of a classifier is given, `initialize` parameter should be set to False.
        If a classifier Class is given, then it will be instanced with `classifier_opts` as initializer values.
    train_data: numerical 2 dimensional ndarray, optional
        Corresponds to the training data features. If none are given, it is assumed that the classifier is
        an instance of a classifier that is already trained. If this parameter is not None and no `train_classes`
        are given, a ValueError is thrown. 
    test_data: numerical 2 dimensional ndarray, optional
        Corresponds to the testing features. This will be given to the `predict` method of the given classifier.
    train_classes: integer ndarray, optional
        Corresponds to the classification of the given `train_data`. It is given to the `fit` method on the 
        classifier. Must be given if `test_data` is not None.
    initialize: bool, optional
        Wether the given `classifier` should be initialized or not. If `classifier` is an instance of a classifier, 
        this value should be set to False. In case `train_data` is None, this value is ignored, and `classifier` is
        supposed to be a trained instance. Default value is True, ie. 'classifier' is interpreted by default as a
        Class object and not an instance. 
    fit_method: string, optional
        The name of the training method on the classifier. Default is 'fit'.
    predict_method: string, optional
        The name of the classifying method on the classifier. Default is 'predict'.
    classifier_opts: dictionary, optional
        Represents the keyword arguments used to instantiate the classifier. Only useful if `classifier` is a Class
        object and `initialize` is set to True.

    Returns
    -------
    predicted_values: ndarray
        Represents the predicted classes of `test_data`. If no testing is performed, this will be an empty array.
    classifier_instance: instance of a classifier
        If `classifier` was an instance, this returns the same object. If `classifier` was a classifier Class object,
        an instance of this class is returned. If `train_data` was not None, then this object will be trained with 
        the given data.
    Examples
    --------
    ( TODO )
    '''

    if not hasattr(classifier, fit_method) or not hasattr(classifier, predict_method):
        raise ValueError(
            f"`classifier` must define methods `{fit_method}` and `{predict_method}` in order to be considered a valid classifier"
        )

    classifier_opts = classifier_opts or dict()

    if initialize:
        classifier_instance = classifier(**classifier_opts)
    else:
        classifier_instance = classifier

    if train_data is not None:
        if train_classes is None:
            raise ValueError(
                "`train_features` given but no `classification` given to train classifier"
            )

        classifier_instance = classifier(**classifier_opts)
        getattr(classifier_instance, fit_method)(train_data, train_classes)

    if test_data is not None:
        prediction = getattr(classifier_instance, predict_method)(test_data)
        return np.array(prediction), classifier_instance

    return np.empty(0), classifier_instance
