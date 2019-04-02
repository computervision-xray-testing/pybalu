__all__ = ['ImageSet']

import os
import re
import collections
import numpy as np
from .imread import imread


class ImageSet:
    def __init__(self, path, *, extension=None, prefix=None, start=None, stop=None,
                 flatten=True,  imloader=None, imloader_opts=None, filenames=None):

        if isinstance(path, ImageSet):
            other = path
            self._directory = other._directory
            self._file_re = re.compile(other._file_re.pattern)
            self._imloader_opts = dict(other._imloader_opts)
            self._flatten = other._flatten
            self._imloader = other._imloader
            if filenames is not None:
                self._filenames = list(filenames)
                self._slice = slice(len(filenames))
                return

            if start is not None or stop is not None:
                other_start = other._slice.start or 0
                _start = other_start + start if start is not None else other_start
                _stop = other._slice.stop if stop is None else other_start + stop
                self._slice = slice(_start, min(_stop, other._slice.stop))
            else:
                self._slice = other._slice

            if self._flatten:
                self._dir_filenames = None
                self._filenames = other._filenames[start: stop]
            else:
                self._dir_filenames = other._dir_filenames[start: stop]
                self._filenames = list(np.hstack(self._dir_filenames))

            # TODO: should we warn if the resulting set is empty?
            return

        if extension is None:
            extension = '.*?'
        else:
            extension = extension.split('.')[-1]

        if prefix is None:
            prefix = ''

        self._directory = path
        self._file_re = re.compile(fr'{prefix}.*\.{extension}$')
        self._imloader_opts = imloader_opts or dict()
        self._imloader = imloader or imread
        self._flatten = flatten
        if flatten:
            self._dir_filenames = None
            self._filenames = [os.path.join(dirname, filename)
                               for dirname, _, files in os.walk(path)
                               for filename in files
                               if self._file_re.match(filename)][start: stop]
            self._slice = slice(start or 0, len(self._filenames))
        else:
            self._dir_filenames = [[os.path.join(dirname, filename)
                                    for filename in files if self._file_re.match(filename)]
                                   for dirname, _, files in os.walk(path)][start: stop]
            self._dir_filenames = [
                dirs for dirs in self._dir_filenames if len(dirs) > 0]
            self._filenames = list(np.hstack(self._dir_filenames))
            self._slice = slice(start or 0, len(self._dir_filenames))

    def set_loader(self, imloader):
        self._imloader = imloader

    def is_flat(self):
        return self._flatten

    def __iter__(self):
        return ((filename, self._imloader(filename, **self._imloader_opts)) for filename in self._filenames)

    def get_filenames(self):
        return list(self._filenames)

    def get_images(self):
        return list(self.iter_images())

    def iter_images(self):
        if self._flatten:
            return (self._imloader(filename, **self._imloader_opts) for filename in self._filenames)
        else:
            return (np.array([self._imloader(filename, **self._imloader_opts) for filename in _dir]) for _dir in self._dir_filenames)

    def __len__(self):
        if self._flatten:
            return len(self._filenames)
        else:
            return len(self._dir_filenames)

    @property
    def shape(self):
        if self._flatten:
            return (len(self._filenames),)
        else:
            return len(self._dir_filenames), len(self._filenames)

    def __getitem__(self, obj):
        if isinstance(obj, int):
            idx = int(obj)
            if self._flatten:
                return self._imloader(self._filenames[idx], **self._imloader_opts)
            else:
                return np.array([self._imloader(filename, **self._imloader_opts) for filename in self._dir_filenames[idx]])
        elif isinstance(obj, slice):
            _slice = obj
            return ImageSet(self, start=_slice.start, stop=_slice.stop)
        elif isinstance(obj, collections.Iterable):
            filenames = np.array(self._filenames)[obj]
            return ImageSet(self, filenames=filenames)
        else:
            raise TypeError(f"Expected a slice or an int, not {type(obj)}")

    def __repr__(self):
        attrs = [f"path='{self._directory}'",
                 f"pattern='{self._file_re.pattern}'"]
        if self._slice.start is not None:
            attrs.append(f'start={self._slice.start}')
        if self._slice.stop is not None:
            attrs.append(f'stop={self._slice.stop}')
        return f'{self.__class__.__name__}({", ".join(iter(attrs))})'
