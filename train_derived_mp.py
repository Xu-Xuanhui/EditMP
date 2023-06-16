from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models_search
import datasets
from functions_mp import train, test, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger
from trajs_data.generate_traj_data import generate_tcp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random




# torch.backends.cudnn.benchmark = True

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def main():
    args = cfg.parse_args()
    print(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    else:
        args.gpu = '0'
        print("Use GPU: {} for training".format(args.gpu))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    if args.gpu is not None:
        print('init model')

        gen_net = eval('models_search.' + args.gen_model + '.Generator')(args=args)
        ort_net = eval('models_search.' + args.gen_model + '.Orthotics')(args=args)
        dis_net = eval('models_search.' + args.dis_model + '.Discriminator')(args=args)


        gen_net.cuda()
        ort_net.cuda()
        dis_net.cuda()

        args.dis_batch_size = int(args.dis_batch_size)
        args.ort_batch_size = int(args.gen_batch_size)
        args.gen_batch_size = int(args.gen_batch_size)
        args.batch_size = args.dis_batch_size


    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        ort_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ort_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter)
    ort_scheduler = LinearLrDecay(ort_optimizer, args.g_lr, 0.0, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter)

    train_data = datasets.Demostration(args.Expert_Trajs_path)

    print(len(train_data))
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    gen_avg_param = 0

    fixed_z = 0
    start_epoch = 0
    # print(dis_net)

    if args.train:
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        writer = SummaryWriter(args.path_helper['log_path'])
        logger.info(args)
        writer_dict = {
            'writer': writer,
            'train_global_steps': start_epoch * len(train_loader),
            'valid_global_steps': start_epoch // args.val_freq,
        }
        for epoch in range(int(0), int(args.max_epoch)):
            # train_sampler.set_epoch(epoch)
            lr_schedulers = (gen_scheduler, dis_scheduler, ort_scheduler) if args.lr_decay else None
            cur_stage = cur_stages(epoch, args)

            train(args, gen_net, dis_net, ort_net, gen_optimizer, dis_optimizer, ort_optimizer, gen_avg_param,
                  train_loader, epoch, writer_dict,
                  fixed_z, lr_schedulers)
            if epoch % 200 == 0:
                print('Testing', epoch)
                results = test(args, gen_net, dis_net,  epoch)
                path = '/home/robot/Documents/EditMP/trajs_data/results/' + str(epoch)
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save('result_ractangles_standard_' + str(epoch) + '.npy', results)
                generate_tcp('result_ractangles_standard_' + str(epoch) + '.npy', epoch)

    else:
            # for i in range(10):
        epoch = 5000
        results = test(args, gen_net, dis_net, epoch)
        # np.save('result_ractangles_standard_' + str(epoch) + '.npy', results)
        # generate_tcp('result_ractangles_standard_' + str(epoch) + '.npy', epoch)

    # fid stat


if __name__ == '__main__':
    main()

