import copy
from pyexpat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from basic.config import logger, device
from basic.models import LeNetZhuMNIST, Lenet5
import numpy as np
import inversefed
# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise


# def add_delta(model, device="cpu"):
#     with torch.no_grad():
#         # modified 原来那样加是不会改变模型的，得这样
#         state_dict = model.state_dict()
#         for name in state_dict:
#             param = state_dict[name]
#             delta = torch.normal(torch.zeros(param.data.size()), 10).to(device)
#             # delta = torch.zeros_like(param.data, requires_grad=True).to(device)
#             delta.requires_grad = True
#             state_dict[name] = param + delta
#     return state_dict

def get_delta_norm_by_eps(eps, D=1, ca=1, c0=1, I=1600, p=0.5):
    """
    reverse Impl of equation(19)
    """
    I_term = np.power(I, p-1)
    delta_norm = (4 * D * (1-eps))/ca - (c0 * I_term)
    return delta_norm


def get_eps_by_delta_norm(delta_net, D=1, ca=1, c0=1, I=1600, p=0.5):
    """
    Impl of equation(19)
    """
    I_term = np.power(I, p-1)
    delta_norm = torch.norm(torch.stack([torch.norm(p, 2.0).to(device) for p in delta_net.parameters()]), 2.0)
    epsilon_p = 1 - ((ca * delta_norm + ca * c0 * I_term) / (4 * D))
    return delta_norm, epsilon_p


def params_norm(params, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    norm_type = float(norm_type)
    if len(params) == 0:
        return torch.tensor(0.)
    device = params[0].device
    if norm_type == np.inf:
        norms = [p.detach().abs().max().to(device) for p in params]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        all_param_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in params])
        total_norm = torch.norm(all_param_norm, norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm, all_param_norm


def proj_by_norm_(parameters, min_norm, max_norm, norm_type=2):
    # 1. calc grad norm
    total_norm, _ = params_norm(parameters, norm_type=norm_type)

    # 2. calc norm-based scaling factor
    min_norm = torch.Tensor([min_norm]).to(total_norm.device)
    max_norm = torch.Tensor([max_norm]).to(total_norm.device)
    if total_norm < min_norm:
        coef = min_norm / (total_norm + 1e-9)
        # for p in parameters:
        #     p.detach().mul_(0.0).add_(min_norm/len())
    else:
        coef = torch.clamp(max_norm / total_norm, max=1.0)

    # 3. apply grad scaling
    for p in parameters:
        p.detach().mul_(coef.to(p.device))

    return total_norm, coef


def distortion_init(tb_writer, comm_R, client_id, model, l, num_distort_iter,
                    privacy_measure='nfl', optimized_target='sigma', element_wise_rand=True):
    """
    initializing the target distortion variable, return as ori_delta_state_dict
    privacy_measure: [nfl, dpl, dpn]
    optimized_target: [variance, value]
    method_type:
        nfl-fix: nfl + value, delta ~ Lap(l_k), scaling delta_norm => l_k
        nfl-learn: nfl + value, delta ~ Lap(l_k), scaling delta_norm => l_k
        dpl-nfl-value: dp + value, delta ~ Lap(l_k=sigma_dp)
        dpl-nfl-variance: dp + variance, delta=l_k=sigma_dp
    """
    assert privacy_measure in ['dp', 'nfl']
    assert optimized_target in ['val', 'sigma']
    ori_delta_state_dict = copy.deepcopy(model.state_dict())
    m = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(l*1.0))
    norm_type = 1 if privacy_measure == 'dp' else 2

    # 1. initializing delta
    for k, v in ori_delta_state_dict.items():
        if '.bn' not in k:
            if element_wise_rand:
                delta = m.sample(v.shape).to(v.device)
            else:
                delta = m.sample([1]).to(v.device)
            ori_delta_state_dict[k].data.fill_(1.0).mul_(delta)
    init_norm, _ = params_norm([v for k,v in ori_delta_state_dict.items() if '.bn' not in k], norm_type=norm_type)
    tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_init', init_norm, comm_R * num_distort_iter)

    # 2. scaling delta norm (except for nfl-dp-val)
    if not (privacy_measure=='dp' and optimized_target=='val') and init_norm != l:
        coef = l / (init_norm + 1e-9)
        for k in ori_delta_state_dict:
            if '.bn' not in k:
                ori_delta_state_dict[k].data.mul_(coef)
        init_norm_scaled, _ = params_norm([v for k,v in ori_delta_state_dict.items() if '.bn' not in k], norm_type=norm_type)
        tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_init_scaled', init_norm_scaled, comm_R * num_distort_iter)
    return ori_delta_state_dict, init_norm


def distortion_learning(tb_writer, comm_R, client_id, batch_data, batch_label, model, CE_criterion, u_loss_type='gap', raw_loss_val=0,
                        num_distort_iter=1, zeta=0.001, lba=10, u=12.0, l=0.,
                        privacy_measure='nfl', optimized_target='val', element_wise_rand=True, dp_upratio=2):
    ori_delta_state_dict, init_norm = distortion_init(tb_writer, comm_R, client_id, model, l, num_distort_iter,
                                                      privacy_measure, optimized_target, element_wise_rand)    
    norm_type = 1 if optimized_target == 'sigma' else 2
    
    if comm_R % 499 == 0:
        delta_stat = {}
        for k,v in ori_delta_state_dict.items():
            layer_size = torch.ones_like(v.detach()).sum().item()
            m, std, norm = v.detach().mean().item(), v.detach().std().item(), torch.norm(v.detach(), norm_type).item()
            delta_stat[k] = dict(mean=m, std=std,norm_avg=norm/layer_size)
            for kk,vv in delta_stat[k].items():
                tb_writer.add_scalar(f'{client_id}_{kk}/{k}_before', vv, comm_R * num_distort_iter)
            tb_writer.add_histogram(f'{client_id}_hist/{k}_before', v.detach().cpu().numpy(), comm_R * num_distort_iter, bins=20)
        

    if privacy_measure == 'dp' and optimized_target == 'val':
        l, u = init_norm, dp_upratio*init_norm  # reevised from sigma --> L1/sum of sampled value
        # tb_writer.add_scalar(f'C{client_id}/nfl_l_4dp', l, comm_R)
        # tb_writer.add_scalar(f'C{client_id}/nfl_u_4dp', u, comm_R)
    
    # TODO: nfl优化过程的Impl可以进一步优化，比如：使用Adam优化器，模型复用同一个
    for iter in range(num_distort_iter):
        # 1. utility loss (by combining net)
        delta_net = copy.deepcopy(model)
        delta_optim = optim.SGD(delta_net.parameters(), lr=zeta)
        delta_state_dict = delta_net.state_dict()
        for name, delta_name in zip(delta_state_dict, ori_delta_state_dict):
            if '.bn' not in name:
                delta = ori_delta_state_dict[delta_name]
                if optimized_target == 'sigma':  # re-parameterization trick of laplacian
                    delta = torch.abs(delta) * laplace_noise(delta.shape, 1, delta.device)
                delta_state_dict[name] += delta
        delta_net.load_state_dict(delta_state_dict)

        pred = delta_net(batch_data)
        if u_loss_type == 'gap':
            utility_loss = torch.square(CE_criterion(pred, batch_label) - raw_loss_val)
        else:
            utility_loss = CE_criterion(pred, batch_label)
        loss = utility_loss

        # 2. privacy loss
        # 替换delta网络参数为原始的delta，以便求导.
        # delta_net.load_state_dict(ori_delta_state_dict)
        if lba != 0:
            for key, param in delta_net.named_parameters():
                if '.bn' not in key:
                    param.data = ori_delta_state_dict[key].data
            delta_norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in delta_net.parameters()]), norm_type)
            dummy_privacy_budget = -delta_norm  # simplified
            
            loss += float(lba) * dummy_privacy_budget
            tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm', delta_norm.item(), comm_R * num_distort_iter + iter)

        # 3.1 update delta
        loss.backward()
        nfl_grad_norm, _ = params_norm([p.grad for p in delta_net.parameters()], norm_type=2)
        delta_optim.step()

        # 3.2 delta norm projection
        norm_type = 2 if privacy_measure == 'nfl' else 1
        total_norm_old, coef = proj_by_norm_(list(delta_net.parameters()), l, u, norm_type=norm_type)
        delta_norm_clipped, _ = params_norm(list(delta_net.parameters()), norm_type=norm_type)
        ori_delta_state_dict = delta_net.state_dict()
        
        # 4. record metrics
        # tb_writer.add_scalar(f'C{client_id}/nfl_delta_layer_norm_mean', all_param_norm.mean(), comm_R * num_distort_iter + iter)
        # tb_writer.add_scalar(f'C{client_id}/nfl_delta_layer_norm_std', all_param_norm.std(), comm_R * num_distort_iter + iter)
        if lba !=0:
            tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm', delta_norm.item(), comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_grad_norm', nfl_grad_norm, comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_clipped', delta_norm_clipped, comm_R * num_distort_iter + iter)
        # tb_writer.add_scalar(f'C{client_id}/nfl_delta_scaling', coef, comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_u_loss', utility_loss.item(), comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_total_loss', loss.item(), comm_R * num_distort_iter + iter)

    if comm_R % 499 == 0:
        for k,v in ori_delta_state_dict.items():
            layer_size = torch.ones_like(v.detach()).sum().item()
            m, std, norm = v.detach().mean().item(), v.detach().std().item(), torch.norm(v.detach(), norm_type).item()
            stat_after = dict(mean=m, std=std,norm_avg=norm/layer_size)
            for name, val in stat_after.items():
                tb_writer.add_scalar(f'{client_id}_{name}/{k}_after', val, comm_R * num_distort_iter)
                tb_writer.add_scalar(f'{client_id}_{name}/{k}_gap', val-delta_stat[k][name], comm_R * num_distort_iter)
            tb_writer.add_histogram(f'{client_id}_hist/{k}_after', v.detach().cpu().numpy(), comm_R * num_distort_iter, bins=20)

    return ori_delta_state_dict


def gaussian_noise(data_shape, sigma, device=None):
    return torch.normal(0, sigma, data_shape).to(device)


def laplace_noise(data_shape, scale, device=None):
    m = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(scale))
    return m.sample(data_shape).to(device)


def dp_scale_laplace(eps, clip, lr):
    sens = 2 * clip * lr
    scale = sens / eps
    return scale


class Client:

    def __init__(self, client_id, ds_name, arch, trainset, valset, testset, shuffle=False, apply_distortion=True, distortion_iter=5,
                 local_batch_iter=1, model_optim="adam", zeta=0.05, lr=3e-4, bs=8, wd=0, le=10, device="cpu",
                 tb_writer=None):
        super(Client, self).__init__()

        assert tb_writer is not None
        self.tb_writer = tb_writer

        self.id = client_id
        self.device = device
        self.apply_distortion = apply_distortion
        self.model_optim = model_optim.lower()

        self.trainset = trainset
        pin_memory = False
        self.trainloader = DataLoader(trainset, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)
        self.valloader = DataLoader(valset, batch_size=bs * 50, shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.testloader = DataLoader(testset, batch_size=bs * 50, shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.train_size = len(trainset)
        self.val_size = len(valset)
        self.test_size = len(testset)

        self.zeta = zeta

        self.local_lr = lr
        self.weight_decay = wd
        self.local_epoch = le
        self.model_optimizer = None

        self.local_batch_iter = local_batch_iter
        self.distortion_iter = distortion_iter

        logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d" % (
            self.id, self.train_size, self.val_size, self.test_size))
        logger.info("local_batch_iter:%2d, distortion_iter:%2d." % (self.local_batch_iter, self.distortion_iter))

        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.model = None

        self.accum_grad_list = list()

        self.init_net(ds_name, arch)

    def get_copied_model(self):
        return copy.deepcopy(self.model)

    def frozen_net(self, frozen):
        for param in self.model.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.model.eval()
        else:
            self.model.train()

    def init_net(self, ds_name, arch):
        """frozen all models' parameters, unfrozen when need to train"""
        num_channels=1 if 'mnist' in ds_name else 3
        model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=num_channels)

        self.model = model
        self.frozen_net(True)

        if self.model_optim == "adam":
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr,
                                              weight_decay=self.weight_decay)
        elif self.model_optim == "sgd":
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        else:
            raise Exception("Does not support {} optimizer for now.".format(self.model_optim))

        self.model.to(device)

    def local_test(self, return_count=False):
        self.frozen_net(True)
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.testloader):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item() if not return_count else (correct, total)

    def local_val(self, return_count=False):
        self.frozen_net(True)
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.valloader):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item() if not return_count else (correct, total)

    # def privacy_leakage(self, params):
    #     d_sum = 0.0
    #     for param in params:
    #         print("param:", param)
    #         # for v in param:
    #         d_sum += param.pow(2).sum()
    #
    #     d_norm = torch.sqrt(d_sum)
    #     print("d_norm", d_norm)
    #     return 1 - (d_norm + 0.1) / 2

    def perform_dp_train(self, x, y, comm_R, clip=12., mechanism='laplace', eps=5, clip_level='sample', element_wise_rand=True):
        assert clip_level in ['sample', 'batch']
        assert mechanism in ['laplace', 'gaussian']
        if mechanism == 'gaussian':
            raise NotImplementedError('dp-sgd+gaussian to be impl')
        clip_norm_type = 1 if mechanism == 'laplace' else 2
        loss_val_list = list()
        acc_val_list = list()
        grad_list = []
        noisy_grad_list = []

        sample_wise_CE = nn.CrossEntropyLoss(reduction='none').to(device)

        clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        for i in range(self.local_batch_iter):
            self.frozen_net(False)
            self.model_optimizer.zero_grad()
            pred = self.model(x)
            loss_samples = sample_wise_CE(pred, y)
            loss_mean = loss_samples.mean()

            # sava raw grad
            client_grad = torch.autograd.grad(loss_mean, self.model.parameters(), retain_graph=True)
            original_grad = list((g.detach().clone() for g in client_grad))
            grad_list.append(original_grad)

            ## 1. clip
            if clip_level == 'sample':  # 单样本缩放
                grad_norms = torch.Tensor([0.0]).to(pred.device)
                for i in range(loss_samples.size()[0]):
                    loss_samples[i].backward(retain_graph=True)
                    grad_norms += torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                 max_norm=clip, norm_type=clip_norm_type)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad
                    self.model.zero_grad()
                grad_norms /= loss_samples.size()[0]
            else:
                loss_mean.backward(retain_graph=True)
                grad_norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                            max_norm=clip, norm_type=clip_norm_type)
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += param.grad
                self.model.zero_grad()

            self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', grad_norms, comm_R)

            ## 2. add noise (gaussian or laplace)
            sens = 2 * clip * self.local_lr
            scale = sens / eps
            self.tb_writer.add_scalar(f'C{self.id}/dp_scale', scale, comm_R)
            for name, param in self.model.named_parameters():
                if element_wise_rand:
                    noise = laplace_noise(clipped_grads[name].shape, scale, device=param.device)
                else:
                    noise = laplace_noise([1], scale, device=param.device) * torch.ones_like(clipped_grads[name])
                clipped_grads[name] += noise
                param.grad = clipped_grads[name]

            # 按照相同的顺序生成noisy_grad并保存，方便DLG使用
            step_noisy_grad = [p.grad.detach().clone() for p in self.model.parameters()]
            noisy_grad_list.append(step_noisy_grad)

            # update local model
            self.model_optimizer.step()

            loss_val = loss_samples.mean().item()
            loss_val_list.append(loss_val)
            acc = torch.sum((torch.argmax(pred, dim=1) == y).float()) / x.size(0)
            acc_val_list.append(acc.item())
            self.tb_writer.add_scalar(f'C{self.id}/train_loss', loss_val, comm_R)
            self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc, comm_R)

            self.frozen_net(True)
        return loss_val_list, acc_val_list, grad_list, noisy_grad_list

    def perform_nfl_train(self, x, y, comm_R, l, u, warming_up=False, nfl_lba=10.,
                        clipDP=-1, u_loss_type='direct', privacy_measure='nfl', optimized_target='val',
                        element_wise_rand=True, dp_upratio=2):
        assert optimized_target in ['val', 'sigma']
        if self.distortion_iter > 0:
            assert u_loss_type in ['gap', 'direct']
        loss_val_list = list()
        acc_val_list = list()
        grad_list = []
        noisy_grad_list = []
        for i in range(self.local_batch_iter):
            self.frozen_net(False)

            self.model_optimizer.zero_grad()
            pred = self.model(x)
            loss = self.CE_criterion(pred, y)

            # sava raw grad
            client_grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            original_grad = list((g.detach().clone() for g in client_grad))
            grad_list.append(original_grad)
            loss.backward()
            
            if clipDP > 0:
                raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipDP, norm_type=1)
                self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', raw_norm, comm_R)
            self.model_optimizer.step()

            loss_val = loss.item()
            loss_val_list.append(loss_val)
            acc = torch.sum((torch.argmax(pred, dim=1) == y).float()) / x.size(0)
            acc_val_list.append(acc.item())
            self.tb_writer.add_scalar(f'C{self.id}/train_loss', loss, comm_R)
            self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc, comm_R)

            if not warming_up and self.apply_distortion == 'nfl':
                # ==1. calculate delta (distortion)
                ori_delta_state_dict = distortion_learning(self.tb_writer, comm_R, self.id, x, y, self.model,
                                                           self.CE_criterion,
                                                           u_loss_type=u_loss_type, raw_loss_val=loss_val,
                                                           num_distort_iter=self.distortion_iter, zeta=self.zeta,
                                                           lba=nfl_lba, u=u, l=l,
                                                           privacy_measure=privacy_measure, optimized_target=optimized_target,
                                                           element_wise_rand=element_wise_rand,
                                                           dp_upratio=dp_upratio)

                # ==2. add delta to the original model
                ori_model_state_dict = self.model.state_dict()
                for name, delta_name in zip(ori_delta_state_dict, ori_model_state_dict):
                    if '.bn' not in name:
                        delta = ori_delta_state_dict[delta_name]
                        if optimized_target == 'val':
                            delta_to_add = delta
                        else:
                            # use abs, because scale must be positive
                            delta_to_add = torch.distributions.laplace.Laplace(torch.zeros_like(delta), torch.abs(delta)).sample([1])[0]
                            delta_to_add.to(delta.device)
                        ori_model_state_dict[name] = ori_model_state_dict[name] + delta_to_add
                        # print("==> distorted:", d_model_state_dict[delta_name])
                self.model.load_state_dict(ori_model_state_dict)

                # ==3. compute grad to distorted model
                # save grad
                noisy_pred = self.model(x)
                noisy_loss = self.CE_criterion(noisy_pred, y)
                noisy_grad = torch.autograd.grad(noisy_loss, self.model.parameters(), retain_graph=True)
                _noisy_grad_ = list((g.detach().clone() for g in noisy_grad))
                noisy_grad_list.append(_noisy_grad_)
                self.model.zero_grad()

            self.frozen_net(True)
        return loss_val_list, acc_val_list, grad_list, noisy_grad_list