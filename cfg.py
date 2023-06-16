

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--max_epoch', type=int, default=40000,
                        help='number of epochs of training')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='set the max iteration number')
    parser.add_argument('-gen_bs', '--gen_batch_size', type=int, default=64,
                        help='size of the batches')
    parser.add_argument('-dis_bs', '--dis_batch_size', type=int, default=64,
                        help='size of the batches')
    parser.add_argument('--g_lr', type=float, default=0.0002,
                        help='adam: gen learning rate')
    parser.add_argument('--wd', type=float, default=0,
                        help='adamw: gen weight decay')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                        help='adam: disc learning rate')
    parser.add_argument('--ctrl_lr', type=float, default=3.5e-4,
                        help='adam: ctrl learning rate')
    parser.add_argument('--lr_decay', action='store_true',
                        help='learning rate decay or not')
    parser.add_argument( '--beta1', type=float, default=0.0,
                         help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--num_workers', type=int,default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=3,
                        help='dimensionality of the latent space')
    parser.add_argument('--n_critic', type=int, default=1,
                        help='number of training steps for discriminator per iter')
    parser.add_argument('--val_freq', type=int, default=20,
                        help='interval between each validation')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='interval between each verbose')
    parser.add_argument('--load_path', type=str,
                        help='The reload model path')
    parser.add_argument('--exp_name', type=str,
                        help='The name of exp')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset type')
    parser.add_argument('--data_path',type=str,default='./data',
                        help='The path of data set')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    parser.add_argument('--gen_model', type=str,
                        help='path of gen model')
    parser.add_argument('--dis_model', type=str,
                        help='path of dis model')
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument('--random_seed', type=int, default=12345)

    # search
    parser.add_argument('--max_search_iter', type=int, default=90,
                        help='max search iterations of this algorithm')
    parser.add_argument('--hid_size', type=int, default=100,
                        help='the size of hidden vector')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--loss', type=str, default="hinge",
                        help='loss function')
    parser.add_argument('--n_classes', type=int, default=0,
                        help='classes')
    parser.add_argument('--grow_steps', nargs='+', type=int,
                        help='the vector of a discovered architecture')
    parser.add_argument('--d_depth', type=int, default=7,
                        help='Discriminator Depth')
    parser.add_argument('--g_depth', type=str, default="5,4,2",
                        help='Generator Depth')
    parser.add_argument('--g_norm', type=str, default="ln",
                        help='Generator Normalization')
    parser.add_argument('--d_norm', type=str, default="ln",
                        help='Discriminator Normalization')
    parser.add_argument('--g_act', type=str, default="gelu",
                        help='Generator activation Layer')
    parser.add_argument('--d_act', type=str, default="gelu",
                        help='Discriminator activation layer')
    parser.add_argument('--diff_aug', type=str, default="None",
                        help='differentiable augmentation type')
    parser.add_argument('--d_heads', type=int, default=4,
                        help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio')
    parser.add_argument('--ema', type=float, default=0.995,
                        help='ema')
    parser.add_argument('--latent_norm',action='store_true',
                        help='latent vector normalization')
    parser.add_argument('--g_mlp', type=int, default=4,
                        help='generator mlp ratio')
    parser.add_argument('--d_mlp', type=int, default=4,
                        help='discriminator mlp ratio')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train or test')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of track actions')
    parser.add_argument('--Expert_Trajs_path', type=str, default='D:/726文件/共享/code/MPGAN/trajs_data/test.npy',
                        help='Expert track storage path')
    opt = parser.parse_args()

    return opt
