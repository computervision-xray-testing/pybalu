__all__ = ['performance']


def performance(real, ideal):
    '''\
    performance(real, ideal)

    (TODO)
    '''
    real = real.squeeze()
    ideal = ideal.squeeze()

    if real.ndim != 1 or ideal.ndim != 1:
        raise ValueError(
            'Cannot compare classifications with more than one dimension')

    return (real == ideal).sum() / real.size
