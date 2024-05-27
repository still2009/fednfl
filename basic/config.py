import os
import argparse
from argparse import Namespace
import logging
import datetime
import torch
import inversefed
import time
import yaml
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # 0,1,4,5,6

def extract_nfl_config(args):
    nfl_cfg_str = args.nfl
    final_cfg_str = '{' + nfl_cfg_str.replace('=', ': ') + '}'
    cfg_dict = yaml.load(final_cfg_str, yaml.Loader)
    nfl_cfg = Namespace(**cfg_dict)

    # base fl parameters
    args.model_optim = cfg_dict.get('model_optim', 'adam')
    args.lr = cfg_dict.get('lr', 0.01)
    args.data_per_client = cfg_dict.get('data_per_client', 1000)
    args.batch_size = cfg_dict.get('B', 8)
    args.local_batch_iter = cfg_dict.get('local_batch_iter', 1)  # #local_iter
    args.shuffle = cfg_dict.get('shuffle', True)

    # dlg parameters
    nfl_cfg.dlg_attack_epochs = list(map(int, cfg_dict.get('dlg_attack_epochs', '0').split('-')))
    nfl_cfg.apply_dlg = cfg_dict.get('dlg', False)
    nfl_cfg.dlg_know_grad = cfg_dict.get('known_grad', 'raw')  # equiv / raw
    nfl_cfg.label_guess = cfg_dict.get('label_guess', True)  # True/False
    nfl_cfg.cost_fn = cfg_dict.get('cost_fn', 'sim')
    nfl_cfg.dlg_optim = cfg_dict.get('dlg_optim', 'lbf')
    nfl_cfg.dlg_img_init = cfg_dict.get('dlg_img_init', 'randn')
    nfl_cfg.dlg_iter = cfg_dict.get('dlg_iter', 1600)  # 900, 1600, 2500
    nfl_cfg.dlg_lr = cfg_dict.get('dlg_lr', 0.01)
    nfl_cfg.tv_lambda = float(cfg_dict.get('tv_lambda', 1e-5))

    # nfl/dp parameters
    nfl_cfg.element_wise_rand = cfg_dict.get('element_wise_rand', True)
    nfl_cfg.dp_upratio = cfg_dict.get('dp_upratio', 2)
    nfl_cfg.warm_up_rounds = cfg_dict.get('warm_up_rounds', 8)
    nfl_cfg.u_loss_type = cfg_dict.get('u_loss_type', 'direct')  # direct, gap
    nfl_cfg.privacy = cfg_dict.get('privacy', 'nfl')  # nfl, dp
    nfl_cfg.opt_target = cfg_dict.get('opt_target', 'val')  # val, sigma
    nfl_cfg.clipDP = float(cfg_dict.get('clipDP', -1))  # no, nfl, dp-gaussian, dp-laplace
    nfl_cfg.apply_distortion = cfg_dict.get('distort', 'no')  # no, nfl, dp-gaussian, dp-laplace
    if type(nfl_cfg.apply_distortion) == bool:
        nfl_cfg.apply_distortion = 'no'
    nfl_cfg.clip = cfg_dict.get('clip', 12.)  # max_norm for grad clip
    nfl_cfg.clipL = cfg_dict.get('clipL', 'batch')  # max_norm for grad clip
    nfl_cfg.eps = cfg_dict.get('eps', 5.)  # epsilon for DP
    nfl_cfg.D = float(cfg_dict.get('D', 56))  # D
    nfl_cfg.ca = float(cfg_dict.get('ca', 5.6/10))  # Dmin / C = D/10/C
    nfl_cfg.c0 = float(cfg_dict.get('c0', 0.1))  # Dmin / C = D/10/C
    nfl_cfg.zeta = float(cfg_dict.get('zeta', 1e-5))  # distortion lr
    nfl_cfg.lba = cfg_dict.get('lba', 10)  # ratio of privacy loss over utility loss
    nfl_cfg.distortion_iter = cfg_dict.get('nfl_it', 10)  # #distortion_iter

    args.nfl = nfl_cfg
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='exp_name')
    parser.add_argument('--out_dir', type=str, default="./runs")
    parser.add_argument('--dataset', type=str, default="fmnist", choices=["fmnist", "mnist", "cifar10",])
    parser.add_argument('--seed', type=int, default=1, help='seed for initializing training (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 8)')
    parser.add_argument('--early_stop_rounds', type=int, default=40, help='early stop rounds')
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay for optimizer (default: 1e-5)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer (default: 3e-4)")
    parser.add_argument('--n_clients', type=int, default=4, help="the number of clients")
    parser.add_argument('--local_epoch', type=int, default=1, help="the epochs for clients' local task training (default: 20)")
    parser.add_argument('--global_epoch', type=int, default=20, help="the epochs for server's training (default: 20)")
    parser.add_argument('--global_iter_per_epoch', type=int, default=100, help="the number of iteration per epoch for server training (default: 100)")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--v', default=False, action='store_true', help='whether to print to standard output')

    # args for NFL
    parser.add_argument('--nfl', type=str, required=False, default="none", help="nfl custom params")
    
    args = parser.parse_args()
    args = extract_nfl_config(args)
    
    # post specify name and ckpt id
    shuffle_str = 'shuf' if args.nfl.shuffle else 'noShuf'
    if 'dp' in args.nfl.apply_distortion:
        fname = f'{args.name}_{shuffle_str}_' \
                f'C{args.nfl.clip}_{args.nfl.clipL}_eps{args.nfl.eps}_' \
                f'{int(time.time())}'
    else:
        fname = f'{args.name}_{shuffle_str}_' \
                f'C{args.nfl.clip}_eps{args.nfl.eps}_' \
                f'lba{args.nfl.lba}_zeta{args.nfl.zeta}_' \
                f'{int(time.time())}'
    args.checkpoint_dir = os.path.join(args.out_dir, fname)
    return args


def init_logger(args):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    fileHandler = logging.FileHandler(os.path.join(args.checkpoint_dir, 'log.txt'), mode='a')
    fileHandler.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if args.v:
        logger.addHandler(consoleHandler)
    return logger


def init_device(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    return device


args = parse_args()
logger = init_logger(args)
device = init_device(args)
inversefed_setup = inversefed.utils.system_startup(gpu=args.gpu)
inversefed_defs = inversefed.training_strategy('conservative')
