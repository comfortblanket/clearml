from utils.smart_api import smart_api_method


class ConstantValueSetter:
    def __init__(self, value):
        self.value = value
    
    @smart_api_method
    def get_value(self):
        return self.value
