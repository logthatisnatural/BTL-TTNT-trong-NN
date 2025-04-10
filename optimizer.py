from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    # State should be stored in this dictionary
                state = self.state[p]

                # Initialize state if not present
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Update step count
                state["step"] += 1
                t = state["step"]

                # Update first and second moments of the gradients
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

                # Bias correction
                if correct_bias:
                    m_hat = m / (1 - beta1 ** t)  # m_hat = m_t / (1 - beta1^t)
                    v_hat = v / (1 - beta2 ** t)  # v_hat = v_t / (1 - beta2^t)
                else:
                    m_hat = m
                    v_hat = v

                # Update parameters
                update = m_hat / (torch.sqrt(v_hat) + eps)  # update = m_hat / (sqrt(v_hat) + eps)
                p.data.add_(-alpha * update)  # p = p - lr * update

                # Add weight decay after the main gradient-based updates
                if weight_decay != 0:
                    p.data.add_(-alpha * weight_decay * p.data)  # p = p - lr * weight_decay * p

        return loss