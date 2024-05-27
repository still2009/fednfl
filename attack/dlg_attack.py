import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from .dlg_utils import calculate_ssim, calculate_psnr, TVloss
from basic.config import device
import inversefed


def sim_reconstruction_costs(gradients, input_gradient, lr, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""

    indices = torch.arange(len(input_gradient))

    weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'sim':
                # print(trial_gradient[i].shape, input_gradient[i].shape, weights[i].shape)
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += (trial_gradient[i]).pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def rec_loss_dlg(dummy_grad, gt_grad):
    grad_diff = 0
    for gx, gy in zip(dummy_grad, gt_grad):
        grad_diff += ((gx - gy) ** 2).sum()
    return grad_diff


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)

        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                # print(trial_gradient[i].shape, input_gradient[i].shape, weights[i].shape)
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += (trial_gradient[i]).pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def encrypt_with_dp(module, sigma=0.001, threshold=5, device="cpu"):
    with torch.no_grad():
        for param in module.parameters():
            # clip factor
            # norm_factor = torch.div(torch.max(torch.norm(param.data)), threshold + 1e-6).clamp(min=1.0)
            # param.data /= norm_factor
            param.data += torch.normal(torch.zeros(param.data.size()), sigma).to(device)


def count_feat_mean_std(features):
    n_channel = features.shape[1]
    channel_means = []
    channel_stds = []
    for i in range(n_channel):
        channel_means.append(torch.mean(features[:, i, :, :]).item())
        channel_stds.append(torch.std(features[:, i, :, :]).item())
    print(channel_means, channel_stds)
    return channel_means, channel_stds


def count_data_mean_std(dataset, sample_label):
    channel_means = []
    channel_stds = []
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=dataset_size, shuffle=False, num_workers=0, pin_memory=True)
    iteration = iter(dataloader)
    image, label = next(iteration)
    print("image:", image.shape)
    sample_image = image[label == sample_label]
    print("sample_image:", sample_image.shape)
    for i in range(3):
        channel_means.append(torch.mean(sample_image[:, i, :, :]).item())
        channel_stds.append(torch.std(sample_image[:, i, :, :]).item())
    print(channel_means, channel_stds)
    return channel_means, channel_stds


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def get_parameters(module_list):
    if len(module_list) == 0:
        return None

    n_module = len(module_list)
    param = list(module_list[0].parameters())
    for i in range(1, n_module):
        param = param + list(module_list[i].parameters())
    return param


def perform_dlg(gt_data,
                gt_one_hot_label,
                model,
                gt_dy_dx,
                local_model_lr,
                dlg_lr,
                lambda_tv,
                optim_iters,
                dlg_optim,
                tb_writer, cid, batch_idx, round, gE=0, cost_fn='sim', label_guess=True):
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = cross_entropy_for_onehot if label_guess else CrossEntropyLoss().to(gt_data.device)
    for param in model.parameters():
        param.requires_grad = True

    #######################################
    # prepare dummy data and dummy label
    #######################################

    dummy_data_init = torch.randn(gt_data.size())
    dummy_data = torch.Tensor(dummy_data_init).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.size()[0], 10)).to(device).requires_grad_(True)

    ##############
    # perform DLG
    ##############

    # optimizer for optimizing 'dummy_data'
    target_variables = [dummy_data, dummy_label] if label_guess else [dummy_data] 
    if dlg_optim == 'lbf':
        optimizer = torch.optim.LBFGS(target_variables, lr=dlg_lr)
    elif dlg_optim == 'sgd':
        optimizer = torch.optim.SGD(target_variables, lr=dlg_lr)
    elif dlg_optim == 'adam':
        optimizer = torch.optim.Adam(target_variables, lr=dlg_lr)
    else:
        raise ValueError(f'dlg_optim={dlg_optim} is invalid')

    best_psnr = 0.0
    best_ssim = 0.0
    best_image = None
    history = []
    interval = 20
    mse_min = 100
    for iters in range(optim_iters):
        global_dlg_step = round*optim_iters + iters
        def closure():
            optimizer.zero_grad()
            pred = model(dummy_data)
            # 原来用的是gt label而不是dummy label
            dummy_onehot_label = F.softmax(dummy_label, dim=-1) if label_guess else gt_one_hot_label
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            # rec_loss = 0
            # for gx, gy in zip(dummy_dy_dx, gt_dy_dx):
            #     # rec_loss += ((gx - (-1 / local_model_lr) * gy).pow(2)).sum()
            #     rec_loss += (((- local_model_lr) * gx - gy).pow(2)).sum()

            # me revised
            # rec_loss = rec_loss_dlg(dummy_dy_dx, gt_dy_dx)
            rec_loss = reconstruction_costs([dummy_dy_dx], gt_dy_dx, cost_fn=cost_fn, indices='def', weights='equal')
            # raw impl
            # rec_loss = reconstruction_costs([dummy_dy_dx], gt_dy_dx, lr=-local_model_lr,
            #                                 cost_fn='sim', indices='def', weights='equal')
            tv_loss = TVloss(dummy_data)
            loss = rec_loss + lambda_tv * tv_loss
            loss.backward()
            
            # use sign
            dummy_data.grad.sign_()


            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_total_loss', rec_loss.item(), global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_rec_loss', rec_loss.item(), global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_tv_loss', tv_loss.item(), global_dlg_step)
            return loss

        optimizer.step(closure)

        if mse_min > (gt_data[0] - dummy_data[0]).pow(2).mean().item():
            mse_min = (gt_data[0] - dummy_data[0]).pow(2).mean().item()

        if iters % interval == 0:
            current_loss = closure()
            x_gen_copy = copy.deepcopy(dummy_data)
            psnr_list = list()
            ssim_list = list()
            for gd, xgd in zip(gt_data, x_gen_copy):
                psnr = calculate_psnr(gd, xgd, max_val=2)
                # print("gd shape:", gd.shape)
                # ssim_score = calculate_ssim(gd, xgd).cpu().detach().item()
                ssim_score = calculate_ssim(torch.unsqueeze(gd, 0).cpu(), torch.unsqueeze(xgd, 0).cpu()).cpu().detach().item()
                psnr_list.append(psnr)
                ssim_list.append(ssim_score)
            psnr_mean = np.mean(psnr_list)
            # print("ssim_list:", ssim_list)
            ssim_mean = np.mean(ssim_list)
            psnr_max = np.max(psnr_list)
            ssim_max = np.max(ssim_list)
            psnr_95 = np.quantile(psnr_list, 0.95)
            ssim_95 = np.quantile(ssim_list, 0.95)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_mse_min', mse_min, global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_psnr_mean', psnr_mean, global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_psnr95', psnr_95, global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_psnr_max', psnr_max, global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_ssim_max', ssim_max, global_dlg_step)
            tb_writer.add_scalar(f'C{cid}/dlg_B{batch_idx}_R{gE}_ssim_95', ssim_95, global_dlg_step)

            if psnr_mean > best_psnr:
                best_psnr = psnr_mean
                # best_image = tt(dummy_data[0].cpu())
                # best_image = (dummy_data[0] + 1) / 2
                best_ssim = ssim_mean
                best_image = dummy_data[0]

            # img = dummy_data[0]
            # # print(img.shape)
            # # img = img.permute(1, 2, 0)
            # img = (img + 1) / 2
            # # history.append(tt(img.cpu()))
            # history.append(img.cpu())
    tb_writer.add_images(f'C{cid}/dlg_B{batch_idx}_R{gE}_imgs', dummy_data, global_dlg_step)
    # print("best PSNR: {:.4f}; best SSIM: {:.4f}".format(best_psnr, best_ssim))
    result = {"PSNR": best_psnr, "SSIM": best_ssim, "ref_image": gt_data[0], "best_image": best_image}
    # ref_image = gt_data[0]
    # plot_pair_images([ref_image, best_image], show_fig=show_figure)
    return result


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4,
               use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    # patched_model = model
    if use_updates:
        patched_model_origin = copy.deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            # outputs = patched_model(inputs, patched_model.parameters)
            outputs = patched_model(inputs)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            # outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size])
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)
        # grad = torch.autograd.grad(loss, patched_model.parameters(), retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(),
                                                      patched_model_origin.parameters.items()))

    # ret_params = OrderedDict()
    # if algorithm == 'fedsplit':
    #     for key in patched_model.parameters:
    #         if 'classifier' in key:
    #             ret_params[key] = patched_model.parameters[key]
    #     # for key in ret_params:
    #     #     print(key)
    #
    #     return list(ret_params.values())
    # # for key in patched_model.parameters:
    # #     print(key)

    return list(patched_model.parameters.values())
