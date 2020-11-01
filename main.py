import os
import time
import argparse

from mmdc import MultiModalDeepClustering


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', action='store_true', help='resume training of a model')
    parser.add_argument('--path', type=str, help='path of model', default='')
    parser.add_argument('--workers', type=int, help='number of dataloader workers', default=8)
    parser.add_argument('--device', type=str, help='gpu device (None to use all available)', default=None)
    parser.add_argument('--plot_rate', type=int, help='plot results every how many epochs', default=1)
    parser.add_argument('--root', type=str, help='root of data folders', default='DATA/')

    # general
    parser.add_argument('--dataset', type=str, help='dataset to train on', required=True)
    parser.add_argument('--arch', type=str, help='network architecture', default='resnet18')
    parser.add_argument('--rotnet', action='store_true', help='use rotnet as an auxiliary task')
    parser.add_argument('--sigma', type=float, help='variance (sigma value) of mixture model gaussians', default=0.0)
    parser.add_argument('--k', type=int, help='number of clusters', default=10)
    parser.add_argument('--gmm_means', type=str, help='gmm means initializer', default="one_hot")
    parser.add_argument('--gmm_dim', type=int, help='gmm dim (keep None for gmm_dim=k)', default=None)

    # optimization
    parser.add_argument('--wd', type=float, help='weight decay parameter', default=0.0005)
    parser.add_argument('--lr', type=float, help='learning rate for optimizer', default=0.05)
    parser.add_argument('--momentum', type=float, help='beta parameter for momentum', default=0.9)
    parser.add_argument('--epochs', type=int, help='number of epochs to train', default=400)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', help='at these epochs lr is decayed', default=[])
    parser.add_argument('--lr_decay_gamma', type=float, help='multiply lr by this when applying decay', default=0.2)
    parser.add_argument('--refine_epoch', type=int, help='start refine stage at this epoch', default=350)

    # image transformations
    parser.add_argument('--crop_size', type=int, nargs='+', help='h/w dimension of center/random crops', required=True)
    parser.add_argument('--input_size', type=int, help='h/w dimension of input to model', required=True)
    parser.add_argument('--sobel', action='store_true', help='apply sobel filters to input as pre-processing')
    parser.add_argument('--flip', action='store_true', help='apply random horizontal flips')
    parser.add_argument('--color_jitter', action='store_true', help='use random color jitters in transformations')
    parser.add_argument('--rot_degree', type=float, help='random rotations range', default=0.0)

    return parser


def create_out_dir():
    dir_name = args.arch + '_lr_' + str(args.lr) + '_wd_' + str(args.wd) + "_sigma_" + str(args.sigma)
    if args.rotnet:
        dir_name += '_rot'
    if args.gmm_dim:
        dir_name += '_gmmdim_' + str(args.gmm_dim)
    path = 'out/%s/%s/%s/' % (args.dataset, dir_name, time.time())
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    out_dir = args.path if args.resume else create_out_dir()

    config = ''
    for arg in vars(args):
        config += arg + ': ' + str(getattr(args, arg)) + '\n'

    with open(os.path.join(out_dir, "config"), "w") as handle:
        handle.write(config)

    print(f"Config:\n{config}")

    mmdc = MultiModalDeepClustering(args, out_dir)
    if args.resume:
        print(f"resuming training from: {args.path}")
        mmdc.load_checkpoint(args.path)
    mmdc.train()

    print("finished.")
