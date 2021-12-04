import warnings
warnings.filterwarnings("ignore")

import sys
import pathlib
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from scripts.model import PCL
import numpy as np
#import imageio
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))
from model.nn import ParallelMLP

def compute_sparsity(mu, normed=True):
    # assume couples, compute normalized sparsity
    diff = mu[::2] - mu[1::2]
    if normed:
        norm = torch.norm(diff, dim=1, keepdim=True)
        norm[norm == 0] = 1  # keep those that are same, dont divide by 0
        diff = diff / norm
    return torch.mean(torch.abs(diff))


class Solver(object):
    def __init__(self, args, image_shape, data_loader=None, logger=None, z_dim=None):
        self.output_dir = args.output_dir
        self.data_loader = data_loader
        self.dataset = args.dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.image_shape = image_shape
        self.logger = logger
        self.r_func_type = args.r_func
        if args.time_limit is not None:
            self.time_limit = args.time_limit * 60 * 60
        else:
            self.time_limit = np.inf
        if z_dim is None:
            self.z_dim = args.gt_z_dim
        else:
            self.z_dim = z_dim
        self.nc = args.num_channel
        if len(image_shape) == 3:
            self.image_shape = image_shape[2:3] + image_shape[:2]
        if self.r_func_type == "default":
            self.params = [nn.Parameter(data=torch.ones(1, self.z_dim, requires_grad=True, device=self.device)),
                           nn.Parameter(data=-torch.ones(1, self.z_dim, requires_grad=True, device=self.device)),
                           nn.Parameter(data=torch.zeros(self.z_dim, requires_grad=True, device=self.device)),
                           nn.Parameter(data=torch.zeros(1, requires_grad=True, device=self.device))]
        elif self.r_func_type == 'mlp':
            self.parallel_mlp = ParallelMLP(2, 1, 512, 5, self.z_dim, bn=False).to(self.device)
            self.params = list(self.parallel_mlp.parameters())
        # for adam
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.net = PCL(self.z_dim, self.nc, architecture=args.architecture,
                           image_shape=image_shape).to(self.device)
        self.optim = optim.Adam(
            self.params + list(self.net.parameters()), lr=self.lr,
            betas=(self.beta1, self.beta2))

        self.ckpt_name = args.ckpt_name
        if False and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.log_step = args.log_step
        self.save_step = args.save_step
	
    def r_func(self, h1, h2):
        if self.r_func_type == 'default':
            Q = torch.sum(torch.abs(self.params[0] * h1 + self.params[1] * h2 + self.params[2]), dim=1)
            Qbar = torch.sum(h1**2, dim=1)
            return - Q + Qbar + self.params[3]
        elif self.r_func_type == 'mlp':
            input = torch.cat([h1.unsqueeze(2), h2.unsqueeze(2)], dim=2)
            return self.parallel_mlp(input).squeeze(2).sum(dim=1)


    def train(self, writer):
        self.net_mode(train=True)
        out = False  # whether to exit training loop
        failure = False  # whether training was stopped
        log = open(os.path.join(self.output_dir, 'log.csv'), 'a', 1)
        log.write('negLLH,H(Q),H(Q;P_N),H(Q;P_L),Total\n')
        log_mean_vars = open(
            os.path.join(self.output_dir, 'mean_vars.csv'), 'a', 1)
        log_mean_vars.write('Total,' + ','.join([str(v) for v in np.arange(
            self.z_dim)]) + '\n')
        log_var_means = open(
            os.path.join(self.output_dir, 'var_means.csv'), 'a', 1)
        log_var_means.write('Total,' + ','.join([str(v) for v in np.arange(
            self.z_dim)]) + '\n')
        log_other = open(os.path.join(self.output_dir, 'other.csv'), 'a', 1)
        log_other.write('|L1|,rate_prior,H(Q_0),H(Q_1)\n')

        # timing
        t0 = time.time()

        while not out:
            for x in self.data_loader:  # don't use label
                # when using a dataset from ILCM, adapt it to fit slowVAE format
                if self.dataset.startswith("toy"):
                    x, _, _, _, _ = x
                    b, t = x.shape[:2]
                    x_new = torch.zeros((b * 2,) + self.image_shape)
                    x_new[::2] = x[:, -2]  # note that if n_lag > 1 in dataset, we are looking only at t-1 and t, ignoring t-2, t-3, ...
                    x_new[1::2] = x[:, -1]
                    x = x_new
                else:
                    x, _ = x

                x = Variable(x.to(self.device))

                x_recon, mu, logvar = self.net(x)
                # mu shape: Batch x latent_dim
                mean_vars = torch.var(mu, dim=0)
                xtm1 = mu[::2]
                xt = mu[1::2]
                xtm1_shuffle = xtm1[torch.randperm(xtm1.shape[0])]
                logits = torch.cat([self.r_func(xt, xtm1), self.r_func(xt, xtm1_shuffle)])
                labels = torch.cat([torch.ones(xt.shape[0]), torch.zeros(xt.shape[0])]).to(self.device)
                vae_loss = F.binary_cross_entropy_with_logits(logits, labels)
                recon_loss, normal_entropy, cross_ent_normal, cross_ent_laplace = torch.zeros(1), torch.zeros(1), \
                                                                                    torch.zeros(1), torch.zeros(1)
                var_means = torch.zeros_like(mean_vars)

                # logging
                sparsity = compute_sparsity(mu, normed=False)

                self.optim.zero_grad()
                vae_loss.backward()
                self.optim.step()

                self.global_iter += 1

                if self.global_iter % self.log_step == 0:
                    if self.logger is not None:
                        self.logger.log_metrics(step=self.global_iter, metrics={"loss_train": vae_loss.item(),
                                                                           "recon_loss": recon_loss.item(),
                                                                           "sparsity": sparsity.item(),
                                                                           "rate_prior": self.rate_prior.item()})

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter or time.time() - t0 > self.time_limit:
                    out = True
                    break

        return failure

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(
                file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.output_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()
