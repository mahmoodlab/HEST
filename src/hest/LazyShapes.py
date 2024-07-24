from hest.utils import verify_paths

class LazyShapes:
    
    def __init__(self, path: str, name: str, coordinate_system: str):
        verify_paths([path])
        self.path = path
        self.name = name
        self.coordinate_system = coordinate_system
        self.shapes = None
        
    def compute(self):
        if self.shapes is None:
            self.shapes = None