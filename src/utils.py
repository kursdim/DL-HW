import time
import numpy as np
from optimization.gd_optimizer import GD
from optimization.adam_optimizer import Adam
from nn.loss_functions.hinge_loss import hinge_loss

def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')


def gradient_check(x, y, neural_net, eps=1e-3, optim_method='GD', lr=1e-3, alpha1=None, alpha2=None):
    # TODO: Реализуйте проверку градиента
    if optim_method == "Adam":
        optimizer = Adam(neural_net.parameters(), lr=lr,
                         alpha1=alpha1, alpha2=alpha2)
    elif optim_method == "GD":
        optimizer = GD(neural_net.parameters(), lr=lr,
                       alpha1=alpha1, alpha2=alpha2)
    for param in neural_net.parameters():
        loss_function = hinge_loss(neural_net(x), y)
        optimizer.zero_grad()
        loss_function.backward()
        grad_backprop = param.grads.flatten()

        grad_numerical = []
        for i in range(param.params.size):
            j = np.unravel_index(i, param.params.shape)
            param_cp = param.params[j]
            param.params[j] = param_cp + eps
            loss1 = hinge_loss(neural_net(x), y).loss
            param.params[j] = param_cp - eps
            loss2 = hinge_loss(neural_net(x), y).loss
            param.params[j] = param_cp
            grad_numerical.append((loss1 - loss2) / (2 * eps))
        grad_numerical = np.array(grad_numerical)

        diff = np.linalg.norm(grad_numerical - grad_backprop) / (np.linalg.norm(grad_numerical) + np.linalg.norm(grad_backprop))
        if diff > eps:
            print(diff)
            print(eps)
            return 'Ошибка в вычислениях градиента'
    return 'Градиенты считаются правильно'
