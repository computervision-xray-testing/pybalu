__all__ = ['create_process_base_class', 'create_process_meta_class']

import warnings
from pybalu.utils import get_func_info
from operator import itemgetter

def create_process_meta_class(clsname, func_name, short_desc):
    return type(f"{clsname}Meta", 
                (ProcessMeta,), 
                {
                    'process_clsname': clsname,
                    'func_name': func_name,
                    'process_short_desc': short_desc,
                })

def create_process_base_class(clsname, func_name, short_desc):
    metacls = create_process_meta_class(clsname, func_name, short_desc)
    return metacls(f"{clsname}Base", tuple(), dict())

class ProcessMeta(type):
    process_clsname = 'Process'
    func_name = 'process_func'
    process_short_desc = 'processing'

    def __new__(cls, clsname, bases, cls_dict):

        if clsname == f'{cls.process_clsname}Base':
            def _init(self):
                raise Exception(f"Can't instantiate `{cls.process_clsname}Base` directly")

            cls_dict['__init__'] = _init 
            return super().__new__(cls, clsname, bases, cls_dict)
        
        if cls.func_name not in cls_dict:
            raise Exception(f"Must declare a valid feature extraction function as `{cls.func_name}` on "
                            f"the declaration of class `{clsname}`.")
            
        func = cls_dict.pop(cls.func_name)
        fname, argnames, default_kwargs, varkwargs = get_func_info(func)
        kwargnames = default_kwargs.keys()
        first_arg = argnames.pop(0)
        
        def __init__(self, *args, **kwargs):
            if len(args) > len(argnames):
                raise TypeError(f"`{clsname}` takes {len(argnames)} positional arguments "
                                f"but {len(args)} were given")

            unknown_args = set(kwargs.keys()) - set(argnames) - set(kwargnames)
            
            if unknown_args and not varkwargs:
                raise TypeError(f"`{clsname}` got an unexpected keyword argument '{unknown_args.pop()}'")
            
            if kwargs.pop('show', False):
                warnings.warn("`show` is always set to False on Processing classes")
            
            kwargs.pop('labels', None)
            
            self._extractor_args = dict(zip(argnames, args))
            self._extractor_args.update(default_kwargs)
            self._extractor_args.update(kwargs)
            self._getter = lambda x: x
            self._idxs = None
        
        def __call__(self, img, *args, **kwargs):
            argdict = dict(self._extractor_args)
            argdict.update(kwargs)
            argdict.update(zip(argnames, args))
            return self._getter(func(img, **argdict))

        def __repr__(self):
            arg_str = ", ".join(f"{k}: {v}" for k, v in self._extractor_args.items())
            getter_str = f"[{self._idxs}]" if self._idxs is not None else ""
            return f"{self.__class__.__name__}{getter_str}({arg_str})"

        def __getitem__(self, idx_tuple_slice):
            self._idxs = idx_tuple_slice            
            if isinstance(idx_tuple_slice, (tuple, list)):
                self._getter = itemgetter(*idx_tuple_slice)
            elif isinstance(idx_tuple_slice, (int, slice)):
                self._getter = itemgetter(idx_tuple_slice)
            else:
                raise TypeError(f"Indices must be integer, slices or tuples, not {type(idx_tuple_slice)}")
            return self

        __init__.__wrapped__ = func

        cls_dict['__init__'] = __init__    
        cls_dict['__call__'] = __call__
        cls_dict['__repr__'] = __repr__
        cls_dict['__getitem__'] = __getitem__
        cls_dict['__doc__'] = f"Wrapper for the `{fname}` {cls.process_short_desc} function. All function parameters\n" +\
                              f"except for `{first_arg}` can be pre-set, so at execution time only `{first_arg}` has\n" +\
                               "to be given. Refer to the function documentation: \n\n"
        cls_dict['__doc__'] += func.__doc__
        
        return super().__new__(cls, clsname, bases, cls_dict)
    
