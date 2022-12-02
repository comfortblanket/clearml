import inspect


def smart_api_function(func):
    params = inspect.signature(func).parameters

    def new_func(state_param_obj):
        try:
            return func( *(state_param_obj.__dict__[p] for p in params) )
        except KeyError as e:
            raise KeyError(f"Requested value not available: {e}")
    
    return new_func


def smart_api_method(method):
    params = inspect.signature(method).parameters
    
    def new_method(self, state_param_obj):
        gen = (p for p in params)
        next(gen)  # Skip 'self'

        try:
            return method( self, *(state_param_obj.__dict__[p] for p in gen) )
        except KeyError as e:
            raise KeyError(f"Requested value not available: {e}")
    
    return new_method
