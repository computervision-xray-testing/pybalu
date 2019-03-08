__all__ = ['print_features']


def print_features(labels, features):
    max_len = len(max(labels, key=len))
    for k, v in zip(labels, features):
        print(f"{k.ljust(max_len)}: {v: .05f}")
