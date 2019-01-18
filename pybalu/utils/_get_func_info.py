__all__ = ['get_func_info']

import inspect

def get_func_info(func):
    sig = inspect.signature(func)
    params = sig.parameters
    args = [name for name, ptype in params.items() 
            if ptype.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    kwargs = dict((name, ptype.default) 
                  for name, ptype in params.items() 
                  if ptype.default is not inspect._empty)
    admits_varkwargs = any([p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()])
    return func.__name__, args, kwargs, admits_varkwargs