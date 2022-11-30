import inspect


def cwrp(param_dict, func):
    # Short for Call With Requested Params
    # 
    # Calls func with whatever arguments it uses in its signature, from 
    # param_dict. For example:
    # 
    # pd = {
    #     "ab" : 1, 
    #     "cd" : 7, 
    #     "de" : 19, 
    # }
    # 
    # def f(de, ab):
    #     return 2*ab + de
    # 
    # def g(ab, c):
    #     return 3*ab
    # 
    # cwrp(pd, f)  # returns 21
    # cwrp(pd, g)  # throws KeyError (since there is no "c" in pd)
    
    try:
        return func(
            *(param_dict[p] for p in inspect.signature(func).parameters)
        )
    except KeyError as e:
        raise KeyError(f"Requested value not available: {e}") from None
