import numpy as np
import copy
import argparse
from multiprocessing.pool import Pool
import subprocess
import yaml
import shutil
import os

basic_kv_dict = {
        # basic params
        'v': False,
        'gpu': 0,
        'out_dir': './runs',
        'ds_name': 'mnist',
    
        # fl params
        'n_clients': 4,
        'data_per_client': 1000,
        'shuffle': 'false',
        'model_optim': 'sgd',
        'lr': 0.1,
        'B': 4,
        'local_batch_iter': 1,
        'global_epoch': 50,
        'early_stop_rounds': 50,
        

        # attack params(dlg)
        'dlg': 'true',
        'dlg_attack_epochs': '0-1-2-5-9-24-49',
        'known_grad': 'equiv',
        'label_guess': 'false',
        'cost_fn': 'sim',
        'dlg_optim': 'adam',
        'dlg_lr': 0.1,  # 当加噪声时，可以适当变小lr以提高攻击效果
        'dlg_iter': 4000,
        'tv_lambda': 1e-6,  # 当加噪声时，可以适当变大tv_lambda以提高攻击效果

        # defense params(dp/nfl)
        'distort': 'no',
        'clipDP': -1,
        'warm_up_rounds': 0,
        'element_wise_rand': True, 
        'dp_upratio': 2,
    }

def get_cmd(CONFIG):
    gpu = CONFIG['gpu']
    verbose = CONFIG.get('v', False)
    task_id, ds_name, n_clients = CONFIG['task_id'], CONFIG['ds_name'], CONFIG['n_clients']
    epoch = CONFIG['global_epoch']
    early_rounds = CONFIG['early_stop_rounds']
    
    nfl_kv_configs = ','.join('{}={}'.format(k,v) for k,v in CONFIG.items())
    cmd = f"python main.py --dataset={ds_name} --out_dir={CONFIG['out_dir']} --n_clients={n_clients} " \
          f"--global_epoch={epoch} --early_stop_rounds={early_rounds} " \
          f"--name='{task_id}' " \
          f'--nfl="{nfl_kv_configs}" --gpu {gpu} {"--v" if verbose else "> /dev/null"}'
    return cmd


def fedsgd():
    param_dict = {}
    for ds_name in ['mnist', 'fmnist', 'cifar10'][-1:]:
        for known_grad in ['updates']:
            for clipDP in [500]:
                for lr in [0.5, 0.1, 0.05]:
                    for model_optim in ['sgd', 'adam']:
                        custom_dict = {
                            'ds_name': ds_name,
                            
                            'lr': lr,
                            'clipDP': clipDP,
                            'known_grad': known_grad,
                            'model_optim': model_optim,
                        }
                        params = copy.deepcopy(basic_kv_dict)
                        params.update(custom_dict)
                        
                        task_id = f"{ds_name}-fedsgd-{model_optim}-lr{lr}_clipDP{clipDP}_dlg-{known_grad}"
                        params['task_id'] = task_id
                        param_dict[task_id] = params
    return param_list

def dlg():
    param_dict = {}
    for ds_name in ['mnist', 'fmnist', 'cifar10']:
        for known_grad in ['updates']:
            for clipDP in [500]:
                for lr in [0.1]:
                    custom_dict = {
                        'lr': lr,
                        'clipDP': clipDP,
                        'known_grad': known_grad,
                    }
                    kv_dict = copy.deepcopy(basic_kv_dict)
                    kv_dict.update(custom_dict)
                    task_id = f"DLG-{ds_name}-fedsgd-lr{lr}_clipDP{clipDP}_dlg-{known_grad}"
                    kv_dict['task_id'] = task_id
                    param_dict[task_id] = kv_dict
    return param_dict

def raw_dp():
    n_clients = 4
    clipDP, eps_scale = 500, 1e5
    warm_up_rounds = 0
    eps_list = [0.01, 0.1, 1, 10]
    print('# ', eps_list)

    param_dict = {}
    for ds_name in ['mnist', 'fmnist', 'cifar10'][2:]:
        for eps in eps_list:
            custom_dict = {
                # fl params
                # attack params(dlg)
                # defense params(dp/nfl)
                'distort': 'dp-laplace',
                'clipL': 'batch',
                'clipDP': clipDP,
                'eps': eps * eps_scale,
                'warm_up_rounds': warm_up_rounds,
            }
            kv_dict = copy.deepcopy(basic_kv_dict)
            kv_dict.update(custom_dict)

            task_id = '{}-dpl_C{}_warm{}_eps{}_dlg-{}'.format(ds_name, clipDP, warm_up_rounds, eps * eps_scale, kv_dict["known_grad"])
            kv_dict['task_id'] = task_id
            param_dict[task_id] = kv_dict
    return param_dict

def raw_dp_dlg():
    n_clients = 4
    clipDP, eps_scale = 1e5, 1e5
    warm_up_rounds = 0
    eps_list = [100]
    print('# ', eps_list)

    param_dict = {}
    for ds_name in ['mnist', 'fmnist', 'cifar10']:
        for eps in eps_list:
            for known_grad in ['raw', 'updates']:
                custom_dict = {
                    # fl params
                    'lr': 0.1,
                    'global_epoch': 1,
                    
                    # attack params(dlg)
                    # defense params(dp/nfl)
                    'distort': 'dp-laplace',
                    'clipL': 'batch',
                    'clipDP': clipDP,
                    'eps': eps * eps_scale,
                    'warm_up_rounds': warm_up_rounds,

                    'cost_fn': 'sim',
                    'known_grad': known_grad, # equiv, updates
                    'dlg_optim': 'adam',
                    'dlg_lr': 0.1,  # 当加噪声时，可以适当变小lr以提高攻击效果
                    'dlg_iter': 4000,
                    'tv_lambda': 1e-6,  # 当加噪声时，可以适当变大tv_lambda以提高攻击效果
                }
                kv_dict = copy.deepcopy(basic_kv_dict)
                kv_dict.update(custom_dict)
                
                task_id = 'DLG-{}-{}-dpl-C{}-eps{}-tv{}'.format(ds_name, known_grad, clipDP, eps * eps_scale, custom_dict['tv_lambda'])
                kv_dict['task_id'] = task_id
                param_dict[task_id] = kv_dict
    return param_dict

def nfl():
    D, clipNFL = 56, 10
    warm_up_rounds = 0  # 9

    out_dir = 'runs_noES_E11_detailATK_mnist_0508'
    param_dict = {}
    for ds_name, lr, dp_eps_list, nfl_eps_list in [
        ('mnist', 0.1,
         # [0.5, 2, 6, 8, 20, 40, 60, 80, 200, 400, 600, 800],
         [60, 80, 200, 400, 600, 800],
         # [0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.99000, 0.99500, 0.99750, 0.99870, 0.99940, 0.99974, 0.99988, 0.99999]
         [0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.99000, 0.99500, 0.99750, 0.99870, 0.99940, 0.99974]
        ),
        
        ('fmnist', 0.1,
         # [0.5, 2, 6, 8, 20, 40, 60, 80, 200, 400, 600, 800],
         [60, 80, 200, 400, 600, 800],
         # [0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.99000, 0.99500, 0.99750, 0.99870, 0.99940, 0.99974, 0.99988, 0.99999]
         [0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.99000, 0.99500, 0.99750, 0.99870, 0.99940, 0.99974]
        ),  # the same as mnist
        
        ('cifar10', 0.5,
         [1e4, 5e4, 10e4, 20e4, 30e4, 40e4, 50e4, 60e4, 70e4, 80e4, 90e4, 100e4],
         [0.970, 0.975, 0.980, 0.985, 0.99000, 0.99500, 0.99750, 0.99870, 0.99940, 0.99974, 0.99988, 0.99999]
        )  # different from others
    ][:2]:
        clipDP, clipNFL = 500, 10
        dlg_img_init = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-adrec/liwenjie37/others/FedNFL/notebooks/{ds_name}.dlg.init'
        for privacy, opt_target, nfl_it, lba, u_loss_type in [
            ('nfl', 'val', 0, 0, 'none'),  # nfl-fix
            # ('nfl', 'val', 5, 1e-5, 'direct'),  # nfl-learn
            ('nfl', 'val', 10, 1e-5, 'direct'),  # nfl-learn
            # ('nfl', 'val', 10, 1e-5, 'gap'),  # nfl-learn-gap

            ('dp', 'val', 0, 0, 'none'),  # dp-l
            # ('dp', 'val', 5, 1e-5, 'direct'),  # dp-nfl
            ('dp', 'val', 10, 1e-5, 'direct'),  # dp-nfl
            # ('dp', 'val', 10, 1e-5, 'gap')  # dp-nfl-gap

            # ('dp', 'sigma', 10, 1e-5, 'direct'),  # dp-nfl
            # ('dp', 'sigma', 10, 1e-5, 'gap')  # dp-nfl
        ]:
            eps_list = dp_eps_list if privacy == 'dp' else nfl_eps_list
            for eps in eps_list:
                custom_dict = {
                    'ds_name': ds_name,
                    'out_dir': out_dir,
                    
                    # fl params
                    'lr': lr,
                    'warm_up_rounds': warm_up_rounds,
                    'global_epoch': 11+warm_up_rounds,
                    
                    # attack params(dlg)
                    'known_grad': 'updates',
                    'dlg_img_init': dlg_img_init,
                    
                    # common defense params
                    'privacy': privacy,
                    'eps': eps,
                    'distort': 'nfl',
                    'nfl_it': nfl_it,
                    'opt_target': opt_target,
                    
                    # for DP
                    'clipDP': clipDP,
                    
                    # for NFL
                    'u_loss_type': u_loss_type,
                    'zeta': 0.1,
                    'clip': clipNFL,
                    'lba': lba,
                    'D': D,
                    'ca': np.round(D/10.0/clipNFL, 3),
                    'c0': 0.01,
                    'element_wise_rand': True, 
                    'dp_upratio': 2,
                    
                }
                kv_dict = copy.deepcopy(basic_kv_dict)
                kv_dict.update(custom_dict)
                task_id = f'{ds_name}-nfl-{privacy}-{opt_target}-warm{warm_up_rounds}-{u_loss_type}-it{nfl_it}-eps{eps}-lba{lba}-dlg-{kv_dict["known_grad"]}'
                kv_dict['task_id'] = task_id
                param_dict[task_id] = kv_dict
    yaml.dump(param_dict, open(f'{out_dir}.yaml', 'w'))
    return param_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='func name')
    parser.add_argument('--gpu', type=str, required=True, help='device str allocate')
    parser.add_argument('--nproc', type=int, required=True, help='parallel level')
    parser.add_argument('--debug', default=False, action='store_true', help='enable debug mode, run ony one cmd.')
    parser.add_argument('--resubmit', default=False, action='store_true', help='resubmit failed experiments')
    args = parser.parse_args()
    return args

def submit_exp(param):
    subprocess.run(get_cmd(param), shell=True)

def check_failed_exp(param_dict):
    fail_param_dict = copy.deepcopy(param_dict)
    out_dir = next(iter(param_dict.values()))['out_dir']
    fail_dirs = []

    # 1. check by out_dir
    for p in os.listdir(out_dir):  # the out_dir may not contain all directories
        tid = p.split('_noShuf')[0]  # drop the tail timestamp
        if os.path.exists(os.path.join(out_dir, p, 'dlg_result_E49.pkl')):
            del fail_param_dict[tid]
        else:
            fail_dirs.append(os.path.join(out_dir, p))
            print('failed with existing out dir', fail_dirs[-1])

    # 2. check by param_dict
    for tid in fail_param_dict:
        print('failed tid', tid)
    
    if input(f'{len(fail_param_dict)}/{len(param_dict)} failed, delete old dirs and resubmit(y/n)? : ') in 'yY':
        for fpath in fail_dirs:
            shutil.rmtree(fpath)
        return fail_param_dict
    else:
        exit(0)
    
if __name__ == '__main__':
    """
    python experiments.py --config debug.yaml --gpu 0 --nproc 1
    python experiments.py --config nfl --gpu 0,1,2,3 --nproc 10 --debug
    python experiments.py --config nfl --gpu 0,1,2,3 --nproc 10
    python experiments.py --resubmit --config grp_x.yaml --gpu 0,1,2,3 --nproc 10
    """
    
    # 1. generate configs
    args = parse_args()
    
    if args.resubmit:
        assert '.yaml' in args.config
        raw_param_dict = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        param_dict = check_failed_exp(raw_param_dict)
    else:
        if '.yaml' in args.config:
            param_dict = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        else:
            param_dict = eval(args.config)()        
    
    # 2. allocate gpu
    gpu_list = args.gpu.split(',')
    print('# total task num: {}'.format(len(param_dict)))
    for i, (tid, params) in enumerate(param_dict.items()):
        gpu_id = gpu_list[i%len(gpu_list)]
        print(f'{i} - {gpu_id} - {tid}')
        param_dict[tid]['gpu'] = gpu_id
    
    # 3. debug task
    if args.debug:
        print('fast debug enabled')
        tid = input(f'debug which task_id? : ')
        submit_exp(param_dict[tid])
    
    # 4. run tasks in group
    if input(f'execute experiments now (total/nproc: {len(param_dict)}/{args.nproc}={len(param_dict)/args.nproc:.3f} times)? y/n: ') in 'yY':
        p = Pool(min(len(param_dict), args.nproc))
        p.map(submit_exp, list(param_dict.values()), chunksize=1)