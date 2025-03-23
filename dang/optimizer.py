from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter], # Các tham số của mô hình (weights, biases)
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999), # tốc độ cập nhật moment
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
            for p in group["params"]: # lấy p là các tham số(weight, bias)
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary 
                state = self.state[p]
                
                # nếu state chưa khởi tạo do là lần đầu, khởi tạo state
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data) # moment bac 1, tensor full 0
                    state["v"] = torch.zeros_like(p.data) # moment bac 2, tensor full 0
                    state["step"] = 0
                # có rồi thì lấy m, v từ state
                m = state["m"]
                v = state["v"]
                
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                
                # Update first and second moments of the gradients
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # update moment bac 1
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # update moment bac 2
                
                state["step"] += 1
                # Bias correction
                if correct_bias:
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])  
                else:
                    m_hat = m
                    v_hat = v
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # Update parameters
                step_size = alpha / (v_hat.sqrt().add_(eps))  
                p.data.add_(-step_size * m_hat)  
            
                # Add weight decay after the main gradient-based updates. 
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)
                # Please note that the learning rate should be incorporated into this update.

        return loss