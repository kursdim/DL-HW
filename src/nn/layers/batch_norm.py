from nn.module.parameters import Parameters
import numpy as np

class BatchNorm:
    """Реализует Batch norm

    ---------
    Параметры
    ---------
    in_dim : int
        Размерность входного вектора

    eps : float (default=1e-5)
        Параметр модели,
        позволяет избежать деления на 0

    momentum : float (default=0.1)
        Параметр модели
        Используется для обновления статистик
    """

    def __init__(self, in_dim, eps=1e-5, momentum=0.1):
        self.in_dim = in_dim
        self.eps = eps
        self.momentum = 0.1

        self.regime = "Train"

        self.gamma = Parameters((in_dim,))
        self.gamma._init_params()

        self.beta = Parameters(in_dim)

        self.E = np.zeros(in_dim)
        self.D = np.zeros(in_dim)

        self.inpt_hat = None
        self.tmp_D = None
        self.tmp_E = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_in)
            Выход слоя
        """
        if self.regime == "Eval":
            # TODO: Реализовать batch norm в eval фазе
            self.inpt_hat = (inpt - self.E) / np.sqrt(self.D + self.eps)
            out = self.inpt_hat * self.gamma.params + self.beta.params
            return out

        # TODO: Реализовать batch norm в train фазе
        self.tmp_D = np.var(inpt, axis=0)
        self.tmp_E = np.mean(inpt, axis=0)
        self.inpt_hat = (inpt - self.tmp_E) / np.sqrt(self.tmp_D + self.eps)
        out = self.inpt_hat * self.gamma.params + self.beta.params

        self.D = (1 - self.momentum) * self.D + self.momentum * self.tmp_D
        self.E = (1 - self.momentum) * self.E + self.momentum * self.tmp_E

        return out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.gamma, self.beta)

    def _zero_grad(self):
        """Обнуляет градиенты модели"""
        self.gamma.grads = np.zeros(self.gamma.shape)
        self.beta.grads = np.zeros(self.beta.shape)

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        if self.regime == "Eval":
            raise RuntimeError("Нельзя посчитать градиенты в режиме оценки")

        # TODO: Реализовать рассчет градиента в batch norm
        self.beta.grads = np.sum(grads, axis=0)
        self.gamma.grads = np.sum(self.inpt_hat * grads, axis=0)
        dx_hat = self.gamma.params * grads
        x_mn = self.inpt_hat * np.sqrt(self.tmp_D + self.eps)
        d_std = 0.5 / np.sqrt(self.tmp_D + self.eps) * (-1. / (self.tmp_D + self.eps) * np.sum(x_mn * dx_hat, axis=0))
        dx1 = 1. / np.sqrt(self.tmp_D + self.eps) * dx_hat + 2 * x_mn * np.ones((self.inpt_hat.shape[0], self.in_dim)) / self.inpt_hat.shape[0] * d_std
        d_mn = -np.sum(dx1, axis=0)
        dx2 = np.ones((self.inpt_hat.shape[0], self.in_dim)) / self.inpt_hat.shape[0] * d_mn
        input_grads = dx1 + dx2

        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def _eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"BatchNorm(in_dim={self.in_dim}, eps={self.eps})"