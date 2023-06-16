# -*- coding: utf-8 -*-


import logging
import operator
import os
from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from imageio import imsave
from utils.utils import make_grid, save_image
from tqdm import tqdm


# from utils.fid_score import calculate_fid_given_paths
from utils.torch_fid_score import get_fid

# from utils.inception_score import get_inception_scorepython exps/dist1_new_church256.py --node 0022 --rank 0sample

logger = logging.getLogger(__name__)
class ReplayBuffer():
    def __init__(self, max_size=5000):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                # to_return.append(element)
        # for j in range()
            if random.uniform(0, 1) > 0.5:
                i = random.randint(0, len(self.data)-1)
                to_return.append(self.data[i].clone())
                # self.data[i] = element
            else:
                to_return.append(element)
            # else:
            #     if random.uniform(0,1) > 0.5:
            #         i = random.randint(0, self.max_size-1)
            #         to_return.append(self.data[i].clone())
            #         self.data[i] = element
            #     else:
            #         to_return.append(element)
        return Variable(torch.cat(to_return))

def cur_stages(iter, args):
    """
    Return current stage.
    :param epoch: current epoch.
    :return: current stage
    """

    idx = 0
    for i in range(len(args.grow_steps)):
        if iter >= args.grow_steps[i]:
            idx = i + 1
    return idx



def KL_divergence(fake, real):
    logp_fake = F.log_softmax(fake.t(), dim=-1)
    p_real = F.softmax(real.t(), dim=-1)
    # print(logp_fake,p_real)
    # kl_sum = F.kl_div(logp_fake, p_real, reduction='sum')
    kl_mean = F.kl_div(logp_fake, p_real, reduction='mean')
    kl_div = kl_mean
    return kl_div
def test(args, gen_net: nn.Module, dis_net: nn.Module, epoch):

    gen_step = 0
    # train mode
    gen_net.load_state_dict(torch.load('/home/robot/Documents/EditMP/exps/model/' + str(epoch) + 'gen_net.pth'))
    dis_net.load_state_dict(torch.load('/home/robot/Documents/EditMP/exps/model/' + str(epoch) + 'dis_net.pth'))
    # z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))).cuda()
    z = torch.cuda.FloatTensor(np.random.rand(args.gen_batch_size, args.latent_dim)).cuda()
    fake_actions = gen_net(z, 0)
    # ort_actions = ort_net(fake_actions.detach(), epoch)
    # print(fake_actions/3.14 * 180)


    return fake_actions.detach().cpu().numpy()


def train(args, gen_net: nn.Module, dis_net: nn.Module, ort_net: nn.Module , gen_optimizer, dis_optimizer,ort_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers):
    lossCfun = torch.nn.MSELoss()
    writer = writer_dict['writer']
    gen_step = 0
    d_loss = 0
    # train mode
    fake_buffer = ReplayBuffer()
    real_buffer = ReplayBuffer()
    gen_net.train()
    dis_net.train()

    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()

    # ---------------------
    #  Train Discriminator
    # ---------------------


    for iter_idx, actions in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        for _ in range(1):
            # Adversarial ground truths
            if args.gf_dim == 1:
                real_actions = actions.type(torch.cuda.FloatTensor).cuda().unsqueeze(-1)
            else:
                real_actions = actions.type(torch.cuda.FloatTensor).cuda()

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (actions.shape[0], args.latent_dim))).cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            real_validity = dis_net(real_actions)
            real_for_kldiv = real_buffer.push_and_pop(real_validity.detach())
            # print('pass dis_net')
            fake_actions = gen_net(z, epoch)

            # print('pass gen_net', fake_actions.size())
            assert fake_actions.size() == real_actions.size(), f"fake_actions.size(): {fake_actions.size()} real_actions.size(): {real_actions.size()}"

            fake_validity = dis_net(fake_actions.detach())

            if args.loss == 'standard':
                real_label = torch.full((actions.shape[0],), 1., dtype=torch.float, device=real_actions.get_device())
                fake_label = torch.full((actions.shape[0],), 0., dtype=torch.float, device=real_actions.get_device())
                real_validity = nn.Sigmoid()(real_validity.view(-1))
                fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                d_real_loss = nn.BCELoss()(real_validity, real_label)
                d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2

            else:
                raise NotImplementedError(args.loss)

            d_loss.backward()

            dis_optimizer.step()
            dis_optimizer.zero_grad()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)
            # -----------------
            #  Train Orthotics
            # -----------------
            rate = (0.1 * torch.rand(real_actions.shape)).cuda()
            # print(rate)
            mid = (1 - rate) * real_actions + rate * (fake_actions - fake_actions.mean(1).unsqueeze(1)).detach()
            # mid = (1 - rate) * real_actions + rate * fake_actions.detach()
            # print(mid.shape)
            # 训练一次校正网络
            out1 = ort_net(mid, epoch)
            lossC = lossCfun(real_actions, out1)
            ort_optimizer.zero_grad()
            lossC.backward()
            ort_optimizer.step()
        # -----------------
        #  Train Generator
        # -----------------
        # if global_steps % (args.n_critic * args.accumulated_times) == 0:
        #
        #     for accumulated_idx in range(args.g_accumulated_times):
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
        # gen_z = torch.cuda.FloatTensor(np.random.rand(args.gen_batch_size, args.latent_dim)).cuda()
        gen_actions = gen_net(gen_z, epoch)

        fake_validity = dis_net(gen_actions)
        fake_for_kldiv = fake_buffer.push_and_pop(fake_validity)
        loss_kldiv = KL_divergence(fake_for_kldiv, real_for_kldiv) * 50
        ort_actions = ort_net(gen_actions, epoch)
        lossC = lossCfun(ort_actions, gen_actions)
        # cal loss
        loss_lz = torch.tensor(0)
        if args.loss == "standard":
            real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float,
                                    device=real_actions.get_device())
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))

            g_loss = (max(0.1,0.99**(epoch//50))) * nn.BCELoss()(fake_validity.view(-1), real_label) + lossC + loss_kldiv#0.99**(epoch // 5)*


        g_loss.backward()

        # torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        writer.add_scalar('g_loss', g_loss.item(), global_steps)
        gen_step += 1
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [KL loss: %f] [C loss: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), loss_kldiv.item(), lossC.item()))
            del gen_actions
            del real_actions
            del fake_validity
            del real_validity
            del g_loss
            del d_loss


        writer_dict['train_global_steps'] = global_steps + 1

    # adjust learning rate
    if schedulers:
        gen_scheduler, dis_scheduler, ort_scheduler = schedulers
        g_lr = gen_scheduler.step(global_steps)
        o_lr = gen_scheduler.step(global_steps)
        d_lr = dis_scheduler.step(global_steps)
        writer.add_scalar('LR/g_lr', g_lr, global_steps)
        writer.add_scalar('LR/d_lr', d_lr, global_steps)


    if epoch % 20 == 0:
        torch.save(gen_net.state_dict(), '/home/robot/Documents/EditMP/exps/model/' + str(epoch) + 'gen_net.pth')
        torch.save(dis_net.state_dict(), '/home/robot/Documents/EditMP/exps/model/' + str(epoch) + 'dis_net.pth')


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
            p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            del cpu_p

    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

