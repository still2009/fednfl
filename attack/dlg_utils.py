import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def TVloss(x, pow=2):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), pow).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), pow).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def frozen_net(net, module_name_list, frozen):
    for module_name in module_name_list:
        if net[module_name] is not None:
            for param in net[module_name].parameters():
                param.requires_grad = not frozen


def to_train(model_dict):
    for _, m in model_dict.items():
        m.train() if m is not None else _


def to_eval(model_dict):
    for _, m in model_dict.items():
        m.eval() if m is not None else _


def save(state_dict, model_dir, model_name):
    torch.save(state_dict, f"{model_dir}/{model_name}.pkl")


def load(net, model_dir, model_name):
    checkpoint = torch.load(f"{model_dir}/{model_name}.pkl", map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint)


def plot_image(image, file_path=None):
    image = image.permute(1, 2, 0)
    image = image.cpu().detach().numpy()
    image = (image + 1) / 2  # [-1,1]->[0,1]
    plt.imshow(image)
    plt.title("generated")
    plt.axis('off')
    plt.show()

    plt.savefig(file_path)


def save_image(image, dir, file_name):
    image = image.permute(1, 2, 0)
    image = image.cpu().detach().numpy()
    image = (image + 1) / 2  # [-1,1]->[0,1]
    print(image, image.shape)
    print(dir + file_name)
    plt.savefig(image, dir + file_name)


def plot_pair_images(images, file_path=None, show_fig=False):
    titles = ["original", "generated"]
    plt.figure(figsize=(10, 6))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        img = images[i]
        img = img.permute(1, 2, 0)
        # print("img: \n{}".format(img))
        img = (img + 1) / 2
        print(f"{i}, img shape:{img.shape}")
        img = img.cpu().detach().numpy()
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')

    if show_fig:
        plt.show()
    if file_path:
        plt.savefig(file_path)
        print("saved figure : {}".format(file_path))


def image_plt(img):
    plt.figure(figsize=(10, 6))
    img = img.permute(1, 2, 0)
    img = (img + 1) / 2
    print(f"img shape:{img.shape}")
    img = img.cpu().detach().numpy()
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    plt.title("generated")
    plt.axis('off')
    plt.show()


# def cal_psnr(real_image, inversion_image):
#     mse = ((real_image - inversion_image) ** 2).mean()
#     mse = mse.cpu().detach().numpy()
#     psnr = 10 * np.log10(4 / mse)
#     return psnr


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data


def deprocess(data):
    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
    elif NChannels == 3:
        # mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        # sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    else:
        print("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0, :, :, :]).unsqueeze(0))


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    code link: https://github.com/snudatalab/KegNet
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super(DiversityLoss, self).__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        if len(noises.shape) > 2:
            noises = noises.view((layer.size(0), -1))
        # print("layer shape:", layer.shape)
        # print("noises shape:", noises.shape)
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


def select_closest_image(ref_extractor, ref_feature, dataset, label):
    """
    Find image (from the dataset) whose feature is the closest to ref_feature.
    :param ref_extractor:
    :param ref_feature:
    :param dataset:
    :param label:
    :return:
    """
    # cos = nn.CosineSimilarity(eps=1e-6)
    closest_image = None
    closest_image_lbl = None
    closest_dist = 10000.0
    for data_idx in range(len(dataset)):
        img, lbl = dataset[data_idx]
        if label == lbl:
            img = img.expand(1, img.shape[0], img.shape[1], img.shape[2])
            feat = ref_extractor(img)
            # dist = cos(ref_feature, feat).item()
            dist = torch.pow((ref_feature - feat), 2).mean().item()

            if dist < closest_dist:
                closest_dist = dist
                closest_image = img
                closest_image_lbl = label
    # print("here", label, data_idx)
    return closest_image, closest_image_lbl


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    # L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be at least 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = 0.01 ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = 0.03 ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


def get_exp_result_file_name(arg_dict, psnr, ssim):
    method = arg_dict.get("method")
    method = "" if method is None else method
    attacker_net_suffix = arg_dict["attacker_net_suffix"]
    net_name = arg_dict["net_name"]
    dataset_name = arg_dict["dataset_name"]
    using_generator = arg_dict["using_generator"]
    learning_rate = arg_dict["learning_rate"]
    lambda_tv = arg_dict["lambda_tv"]
    lambda_prior = arg_dict["lambda_prior"]
    apply_dp = arg_dict["apply_dp"]
    sigma = arg_dict["sigma"]
    label = arg_dict["label"]
    index = arg_dict["index"]

    args_list = [method, attacker_net_suffix, net_name, dataset_name]
    args_dict = OrderedDict()
    args_dict["gen"] = using_generator
    args_dict["lbl"] = label
    args_dict["idx"] = index
    args_dict["lr"] = learning_rate
    args_dict["tv"] = lambda_tv
    args_dict["pr"] = lambda_prior
    if apply_dp:
        args_dict["sigma"] = sigma
    args_dict["ssim"] = ssim
    args_dict["psnr"] = psnr

    args_list = args_list + [name + str(val) for name, val in args_dict.items()]
    return "-".join(args_list)


def calculate_psnr(real_image, inversion_image, max_val=1):
    mse = ((real_image - inversion_image) ** 2).mean()
    mse = mse.cpu().detach().numpy()
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def get_setting_file_name(arg_dict):
    method = arg_dict.get("method")
    method = "" if method is None else method
    attacker_net_suffix = arg_dict["attacker_net_suffix"]
    net_name = arg_dict["net_name"]
    dataset_name = arg_dict["dataset_name"]
    using_generator = str(arg_dict["using_generator"])
    apply_dp = str(arg_dict["apply_dp"])

    args_list = [method, attacker_net_suffix, net_name, dataset_name, "gen" + using_generator, "dp" + apply_dp]

    return "-".join(args_list)


def average_score(score_list):
    mean = np.mean(score_list)
    std = np.std(score_list)
    return mean, std


def average_psnr_ssim(psnr_list, ssim_list):
    psnr_mean, psnr_std = average_score(psnr_list)
    ssim_mean, ssim_std = average_score(ssim_list)
    return (psnr_mean, psnr_std), (ssim_mean, ssim_std)


def select_image(dataset, label, index):
    count = 0
    for j in range(len(dataset)):
        img, lbl = dataset[j]
        if label == lbl:
            if index == count:
                return img
            count += 1


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target