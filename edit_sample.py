import numpy as np
import torch
import torch.nn as nn
from trajs_data.ur_kinematics_torch import angle2tcp
from trajs_data.ceshi_simulate import *




def edit_width_and_height(args,  gen_net: nn.Module, sty_net: nn.Module ,sty_optimizer,sty_scheduler,  epoch):

    path = 'path for model parameters'  # path for model parameters
    lossL1 = nn.L1Loss()
    gen_net.load_state_dict(torch.load(path + str(epoch) + 'gen_net.pth'))

    w = 200 # expected width
    h = 200 # expected height

    gen_net.eval()
    sty_net.train()

    label_w = torch.ones(1).repeat(args.gen_batch_size, 1).detach().cuda() * w
    label_h = torch.ones(1).repeat(args.gen_batch_size, 1).detach().cuda() * h

    z_seed = torch.cuda.FloatTensor(np.random.normal(0, 1, args.latent_dim))
    z = z_seed.repeat(args.gen_batch_size, 1).detach()

    for t in range(500):
        if t % 50 == 0:
            z_seed = torch.cuda.FloatTensor(np.random.normal(0, 1, args.latent_dim))
            z = z_seed.repeat(args.gen_batch_size, 1).detach()
        Coordinates_ = torch.zeros((args.gen_batch_size, 8, 3)).cuda()

        z_ = sty_net(z)
        fake_actions_ = gen_net(z_, 0)
        for l in range(8):
            Coordinates_[:, l, :] = angle2tcp(fake_actions_[:, l, :]) * 1000

        loss_h = lossL1(Coordinates_[:, 4, 1] - Coordinates_[:, 2, 1], label_h)
        loss_w = lossL1(Coordinates_[:, 2, 0] - Coordinates_[:, 0, 0], label_w)

        loss = loss_h + loss_w
        loss.backward()
        sty_optimizer.step()
        sty_optimizer.zero_grad()
        s_lr = sty_scheduler.step(t * 5)
        print(loss_h , loss_w, loss)
        if loss_w <= 9 and loss_h <= 5:
            break

    return fake_actions_.detach().cpu().numpy()


def edit_target_points(args, z_, gen_net: nn.Module, sty_net: nn.Module ,sty_optimizer,sty_scheduler, x,y,z, epoch):

    path = 'path for model parameters'  # path for model parameters
    success_rate = 0
    lossL1 = nn.L1Loss()
    gen_net.load_state_dict(torch.load(path + str(epoch) + 'gen_net.pth'))

    gen_net.eval()
    sty_net.train()
    label_y1 = torch.ones(args.gen_batch_size).cuda() * y[1].detach()
    label_z1 = torch.ones(args.gen_batch_size).cuda() * z[1].detach()
    label_x2 = torch.ones(args.gen_batch_size).cuda() * x[2].detach()
    label_y2 = torch.ones(args.gen_batch_size).cuda() * y[2].detach()

    for t in range(500):
        # if t == 100:
        #     z_seed = torch.cuda.FloatTensor(np.random.normal(0, 1, args.latent_dim))
        #     # z_selected = screen(args, z_seed, gen_net)
        #     z = z_seed.repeat(args.gen_batch_size, 1).detach()
        Coordinates_ = torch.zeros((args.gen_batch_size, 8, 3)).cuda()
        z__ = sty_net(z_)
        fake_actions_ = gen_net(z__, 0)
        for l in range(8):
            Coordinates_[:, l, :] = angle2tcp(fake_actions_[:, l, :]) * 1000
        # print(Coordinates_[0, 0])
        loss_1 = lossL1(Coordinates_[:, 4, 1], label_y1) + lossL1(Coordinates_[:, 4, 2], label_z1)
        loss_2 = lossL1(Coordinates_[:, 7, 0], label_x2) + lossL1(Coordinates_[:, 7, 1], label_y2)

        loss = loss_1 + loss_2
        if loss/4 <= 10 :
            success_rate = 1
            print("success", loss_1.item(), loss_2.item())
            break

        loss.backward()
        sty_optimizer.step()
        sty_optimizer.zero_grad()
        s_lr = sty_scheduler.step(t * 5)
        if t % 10 ==0:

            print('epoch:', t, 'loss1:', loss_1.item()/8, 'loss2:', loss_2.item()/8)

    return success_rate
