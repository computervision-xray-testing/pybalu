__all__ = ['normalize']

def normalize(X):
    mf = X.mean(0)
    sf = X.std(0)
    a = 1 / sf
    b = - mf / sf
    return X * a + b, a, b