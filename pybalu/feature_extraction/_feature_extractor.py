__all__ = ['FeatureExtractorBase', 
           'FeatureExtractionPipeLine', 
           'FeatureExtractionGrouping',
           'FeatureExtractionMultiplexer']

import inspect
from pybalu.utils import create_process_base_class
import numpy as np

FeatureExtractorBase = create_process_base_class("FeatureExtractor", "extractor_func", "feature extraction")

class FeatureExtractionPipeLine:
    '''\
    (TODO)
    '''
    def __init__(self, *args):
        try:
            *self._preprocessors, self._extractor = args
        except ValueError:
            raise TypeError(f"{self.__class__.__name__} expected at least 1 argument, got 0" )

        for p in self._preprocessors:
            if not callable(p):
                raise TypeError(f"Preprocessors must be callable, received {type(p)} instead")

        if not isinstance(self._extractor, (FeatureExtractorBase, FeatureExtractionGrouping)):
            raise TypeError(f"Last process on pipe must be a FeatureExtractor, not {type(self._extractor)}")

    def __repr__(self):
        inner = ',\n    '.join(str(p) for p in self._preprocessors + [self._extractor])
        return f"{self.__class__.__name__}(\n    " + inner + '\n)'
    
    def __call__(self, img, **kwargs):
        result = img

        for process in self._preprocessors:
            result = process(result)
        if not isinstance(result, tuple):
            result = (result,)
        return self._extractor(*result, **kwargs)

class FeatureExtractionGrouping:
    '''\
    (TODO)
    '''
    def __init__(self, *extractors):
        if len(extractors) == 0:
            raise TypeError(f"{self.__class__.__name__} expected at least 1 argument, got 0" )

        for e in extractors:
            if not isinstance(e, FeatureExtractorBase):
                raise TypeError(f"Expected a group of extractors, received {type(e)} instead")

        self._extractors = extractors

    def __repr__(self):
        inner = ',\n    '.join(str(e) for e in self._extractors)
        return f"{self.__class__.__name__}(\n    " + inner + '\n)'
    
    def __call__(self, img, *args, **kwargs):
        results = [e(img, *args, **kwargs) for e in self._extractors]
        if kwargs.get('labels', False):
            features = np.hstack([feats for _, feats in results])
            labels = np.hstack([l for l, _ in results])
            return labels, features
        return np.hstack(results)


class FeatureExtractionMultiplexer:
    '''\
    (TODO)
    '''
    def __init__(self, extractor):

        if not isinstance(extractor, (FeatureExtractorBase, FeatureExtractionGrouping, FeatureExtractionPipeLine)):
            raise TypeError(f"Expected an extractors, received {type(extractor)} instead")

        self._extractor = extractor

    def __repr__(self):
        return f"{self.__class__.__name__}(\n    {str(self._extractor)}\n)"
    
    def __call__(self, images, *args, **kwargs):
        results = [self._extractor(img, *args, **kwargs) for img in images]
        if kwargs.get('labels', False):
            features = np.hstack([feats for _, feats in results])
            labels = np.hstack([[f"[{i}]{l}" for i, l in enumerate(labels)] for labels, _ in results])
            return labels, features
        return np.hstack(results)