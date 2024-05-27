import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from multiprocessing.pool import Pool
import numpy as np
import torch
import pandas as pd
import re
import datetime as dt
import logging
import pickle as pk


def judge_condition(k, cond_list):
    """
    给出一个条件列表cond_list，判断关键字k是否满足该列表
    cond_list: a list of and_condition, each and_condition is a list of or_condition
    k: a string, typically a file name
    judge rule: whether the condition string is in the key string
    """
    result = True
    for and_cond in cond_list:
        cur_res = False
        for or_cond in and_cond:
            if or_cond in k:
                cur_res = True
                break
        if not cur_res:
            return False
    return True
            
def extract_single_event(fpath, metrics_filter_pats=[['']]):
    # 读取tb event
    e = EventAccumulator(fpath)
    try:
        e.Reload()
    except Exception:
        logging.warning(f'error loading {fpath}')
        return fpath, {}
    df_dict = {k: pd.DataFrame(e.Scalars(k)) for k in e.Tags()['scalars'] if judge_condition(k, metrics_filter_pats)}
    return fpath, df_dict

def extract_multi_events(paths, max_worker=5, metrics_filter_pats=[['']]):
    p = Pool(min(max_worker, len(paths)))
    event_data = p.starmap(extract_single_event, [(p, metrics_filter_pats) for p in paths])
    return dict(event_data)


def val_acc_anaysis(task_data_dict, pattern_str):
    pattern = re.compile(pattern_str)
    # 4.2 val_acc攻击的PSNR分析
    acc_metric = {}
    for k,v_dict in task_data_dict.items():
        local_mean = np.mean([v_dict[f'C{cid}/val_acc'].value.max() for cid in range(4)])
        global_val = v_dict.get('global/val_acc', None)
        global_test = v_dict.get('global/test_acc', None)
        param_str = str(pattern.findall(k)[0])
        acc_metric[param_str] = [local_mean,
                                 -1 if type(global_val)==type(None) else global_val.value.max(),
                                 -1 if type(global_test)==type(None) else global_test.value.max()
                                ]
    return acc_metric

# 基于单独存储的best utility结果 (速度快)
def get_best_metric(task_path_list):
    result = []
    for path in task_path_list:
        if not os.path.exists(os.path.join(path, 'best_metric.txt')):
            continue
        with open(os.path.join(path, 'best_metric.txt'), 'r') as f:
            values = f.readlines()[0].strip().split(',')
            values = list(map(float, values))
            tid = os.path.split(path)[-1]
            result.append(dict(tid=tid, epoch=values[0], global_val_acc=values[1], global_test_acc=values[2]))
    return pd.DataFrame(result)

# 基于单独存储的dlg privacy结果 (速度快)
def get_dlg_results(task_path_list, pattern_str, pat_name, dlg_fname='dlg_result.pkl'):
    pattern = re.compile(pattern_str)
    result = []
    img_result = {}
    for path in task_path_list:
        dlg_result_file = os.path.join(path, dlg_fname)
        if os.path.exists(dlg_result_file):
            dlg_result_dict = pk.load(open(dlg_result_file, 'rb'))
            tid = os.path.split(path)[-1]
            
            # extract img
            img_result[tid] = {k: dlg_result_dict[k] for k in ['rec_img', 'gt']}
            del dlg_result_dict['rec_img']
            del dlg_result_dict['gt']
            
            # extract metrics
            exp_ablation_dict = dict(zip(pat_name, pattern.findall(tid)[0]))
            result.append(dict(tid=tid, **dlg_result_dict, **exp_ablation_dict))
    return pd.DataFrame(result), img_result

# 基于event文件的psnr统一分析 (需要读取全部event文件，内存消耗大)
def psnr_anaysis_from_event(task_data_dict, pattern_str, pat_name):
    pattern = re.compile(pattern_str)
    # 4.1 DLG攻击的PSNR分析
    dlg_results = []
    dlg_metrics = 'mse,feat_mse,psnr,ssim'.split(',')
    for k,v_dict in task_data_dict.items():
        metric_values = np.round([v_dict[f'C0/dlg_B0_{name}'].value.values[-1] for name in dlg_metrics], 3)
        param_str = pattern.findall(k)[0]
        dlg_results.append([k.split('/')[-1], *param_str, *list(metric_values)])
    df_results = pd.DataFrame(dlg_results)
    df_results.columns = ['tid', *pat_name.split(','), *dlg_metrics]
    return df_results