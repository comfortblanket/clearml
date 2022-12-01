import inspect


class SmartAPI:
    def __init__(self):
        self.function_signatures = {}

    def cwrp(self, param_dict, func):
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
        
        if func in self.function_signatures:
            sig = self.function_signatures[func]
        else:
            sig = inspect.signature(func).parameters
            self.function_signatures[func] = sig

        try:
            return func( *(param_dict[p] for p in sig) )
        except KeyError as e:
            raise KeyError(f"Requested value not available: {e}") from None
