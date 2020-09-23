import torch
import torch.nn.functional as F
from copy import deepcopy


class ALR(object):
    def __init__(self, d_X, d_Y, lambda_lp, eps_min, eps_max, xi, ip, K):
        super(ALR, self).__init__()
        self.d_X = d_X
        self.d_Y = d_Y
        self.lambda_lp = lambda_lp
        self.eps_min = eps_min
        self.eps_max = eps_max
        if eps_min == eps_max:
            self.eps = lambda x: eps_min * torch.ones(x.size(0), 1, 1, 1, device=x.device)
        else:
            self.eps = lambda x: eps_min + (eps_max - eps_min) * torch.rand(x.size(0), 1, 1, 1, device=x.device)
        self.xi = xi
        self.ip = ip
        self.K = K

    def virtual_adversarial_direction(self, f, x):
        batch_size = x.size(0)
        f.zero_grad()
        normalize = lambda vector: F.normalize(vector.view(batch_size, -1, 1, 1), p=2, dim=1).view_as(x)
        d = torch.rand_like(x) - 0.5
        d = normalize(d)
        for _ in range(self.ip):
            d.requires_grad_()
            x_hat = torch.clamp(x + self.xi * d, min=-1, max=1)
            y = f(x)
            y_hat = f(x_hat)
            y_diff = self.d_Y(y, y_hat)
            y_diff = torch.mean(y_diff)
            y_diff.backward()
            d = normalize(d.grad).detach()
            f.zero_grad()
        r_adv = normalize(d) * self.eps(x)
        r_adv[r_adv != r_adv] = 0
        r_adv[r_adv == float("inf")] = 0
        r_adv_mask = torch.clamp(
            torch.lt(torch.norm(r_adv.view(batch_size, -1, 1, 1), p=2, dim=1, keepdim=True), self.eps_min).float()
            +
            torch.gt(torch.norm(r_adv.view(batch_size, -1, 1, 1), p=2, dim=1, keepdim=True), self.eps_max).float(),
            min=0, max=1
        ).expand_as(x)
        r_adv = (1 - r_adv_mask) * r_adv + r_adv_mask * normalize(torch.rand_like(x) - 0.5)
        return r_adv

    def get_adversarial_perturbations(self, f, x):
        r_adv = self.virtual_adversarial_direction(f=deepcopy(f), x=x.detach())
        x_hat = x + r_adv
        return x_hat

    def get_alp_loss(self, x, x_hat, y, y_hat):
        y_diff = self.d_Y(y, y_hat)
        x_diff = self.d_X(x, x_hat)
        nan_count = torch.sum(y_diff != y_diff).item()
        inf_count = torch.sum(y_diff == float("inf")).item()
        neg_count = torch.sum(y_diff < 0).item()
        lip_ratio = y_diff / x_diff
        alp = torch.clamp(lip_ratio - self.K, min=0)
        nonzeros = torch.nonzero(alp)
        alp_count = nonzeros.size(0)
        alp_l1 = torch.mean(alp)
        alp_l2 = torch.mean(alp ** 2)
        alp_loss = self.lambda_lp * alp_l1
        return (
            alp_loss, lip_ratio, x_diff, y_diff, alp_l1, alp_l2, alp_count, nan_count, inf_count, neg_count
        )