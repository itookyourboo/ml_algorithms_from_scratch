
class Getter:
    def __init__(self, name: str):
        self.name = name
        self._function = getattr(self, name)

    def get(self, *args):
        return self._function(*args)
