
class ConstantValueSetter:
    def __init__(self, value):
        self.value = value
    
    def get_value(self, sim_state):
        return self.value
