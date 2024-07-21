class MyLineReg:
    def __init__(self, n_iter: int, learning_rate: float):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
