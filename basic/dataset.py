import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from basic.config import args, logger, device


class CustomDataset(Dataset):

    def __init__(self, dataset, transform):
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image = self.transform(image)
        label = self.dataset[idx][1]
        return image, label


def easy_data_partition(n_clients, trainset, testset, shuffle, n_train_data_per_client = 1000):
    trainsets, valsets, testsets = [], [], []
    train_size, test_size = len(trainset), len(testset)
    split_size = int(test_size / n_clients / 2)
    logger.info("train data size:%2d, test data size:%2d, split size:%2d" % (train_size, test_size, split_size))

    train_index, test_index = np.arange(train_size), np.arange(test_size)
    if shuffle:
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
    
    for i in range(n_clients):
        train_data = Subset(trainset, train_index[n_train_data_per_client * i:n_train_data_per_client * (i + 1)])
        val_data = Subset(testset, test_index[split_size * i:split_size * (i + 1)])
        test_data = Subset(testset, test_index[split_size * (i + n_clients):split_size * (i + 1 + n_clients)])
        trainsets.append(train_data)
        valsets.append(val_data)
        testsets.append(test_data)
        logger.info("client id:%2d" % (i))
        print_data_distribution(train_data, val_data, test_data)
    return trainsets, valsets, testsets


def print_data_distribution(train_data, val_data, test_data):
    text = "train data: "
    train_label = Counter([data[1] for data in train_data])
    for j in range(10):
        text += str("%2d:%-4d," % (j, train_label[j]))
    logger.info(text)
    text = "val   data: "
    val_label = Counter([data[1] for data in val_data])
    for j in range(10):
        text += str("%2d:%-4d," % (j, val_label[j]))
    logger.info(text)
    text = "test  data: "
    test_label = Counter([data[1] for data in test_data])
    for j in range(10):
        text += str("%2d:%-4d," % (j, test_label[j]))
    logger.info(text)


class MyDataset(Dataset):
    def __init__(self, path) -> None:
        assert os.path.exists(path)
        self.data, self.targets = torch.load(path)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)

class FMNIST(object):

    def __init__(self):
        pass

    def data_partition(self, shuffle):
        print(f'partition data with shuffle? {shuffle}')
        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # trainset = datasets.FashionMNIST(root="data/fmnist", train=True, download=True, transform=train_transform)
        # testset = datasets.FashionMNIST(root="data/fmnist", train=False, download=True, transform=test_transform)
        trainset = MyDataset('data/fmnist/FashionMNIST/my_preprocessed/train.pt')
        testset = MyDataset('data/fmnist/FashionMNIST/my_preprocessed/test.pt')

        self.trainsets = []
        self.valsets = []
        self.testsets = []

        train_size = len(trainset)
        train_index = np.arange(train_size)
        test_size = len(testset)
        test_index = np.arange(test_size)
        if shuffle:
            np.random.shuffle(train_index)
            np.random.shuffle(test_index)

        split_size = int(test_size / args.n_clients / 2)

        logger.info("train data size:%2d" % (train_size))
        logger.info("test data size:%2d" % (test_size))
        logger.info("split size:%2d" % (split_size))

        for i in range(args.n_clients):
            n_train_data_per_client = 1000
            train_data = Subset(trainset, train_index[n_train_data_per_client * i:n_train_data_per_client * (i + 1)])
            # train_data = Subset(trainset, train_index[2000 * i:2000 * (i + 1)])
            val_data = Subset(testset, test_index[split_size * i:split_size * (i + 1)])
            test_data = Subset(testset,
                               test_index[split_size * (i + args.n_clients):split_size * (i + 1 + args.n_clients)])
            # val_data = Subset(testset, test_index[0:500])
            # test_data = Subset(testset, test_index[-500:])
            self.trainsets.append(train_data)
            self.valsets.append(val_data)
            self.testsets.append(test_data)

            logger.info("client id:%2d" % (i))
            print_data_distribution(train_data, val_data, test_data)

        if args.algorithm == "feddf":
            self.server_dataset = Subset(trainset, train_index[2000 * args.n_clients:2000 * args.n_clients + 32000])
            return self.trainsets, self.valsets, self.testsets, self.server_dataset
        else:
            return self.trainsets, self.valsets, self.testsets


class Cifar(object):
    def __init__(self):
        pass

    def data_partition(self):

        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        trainset = datasets.CIFAR10(root="data/cifar", train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root="data/cifar", train=False, download=True, transform=test_transform)

        self.trainsets = []
        self.valsets = []
        self.testsets = []

        train_size = len(trainset)
        train_index = np.arange(train_size)
        np.random.shuffle(train_index)
        test_size = len(testset)
        test_index = np.arange(test_size)
        np.random.shuffle(test_index)
        split_size = int(test_size / args.n_clients / 2)

        for i in range(args.n_clients):
            train_data = Subset(trainset, train_index[2000 * i:2000 * (i + 1)])
            val_data = Subset(testset, test_index[split_size * i:split_size * (i + 1)])
            test_data = Subset(testset,
                               test_index[split_size * (i + args.n_clients):split_size * (i + 1 + args.n_clients)])
            self.trainsets.append(train_data)
            self.valsets.append(val_data)
            self.testsets.append(test_data)

            logger.info("client id:%2d" % (i))
            print_data_distribution(train_data, val_data, test_data)

        if args.algorithm == "feddf":
            self.server_dataset = Subset(trainset, train_index[2000 * args.n_clients:2000 * args.n_clients + 32000])
            return self.trainsets, self.valsets, self.testsets, self.server_dataset
        else:
            return self.trainsets, self.valsets, self.testsets


class Digit(object):

    def __init__(self):
        pass

    def data_partition(self):

        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainsets = []
        self.valsets = []
        self.testsets = []
        for domain in ["mnist_m", "svhn", "syn", "usps"]:
            trainset = ImageFolder("data/digit/" + domain + "/train_images", train_transform)
            size = len(trainset)
            index = np.arange(size)
            np.random.shuffle(index)
            train_data = Subset(trainset, index[:2000])
            self.trainsets.append(train_data)

            testset = ImageFolder("data/digit/" + domain + "/test_images", test_transform)
            size = len(testset)
            index = np.arange(size)
            np.random.shuffle(index)
            split_size = int(size / 2)
            val_data = Subset(testset, index[:split_size])
            test_data = Subset(testset, index[split_size:])
            self.valsets.append(val_data)
            self.testsets.append(test_data)

            logger.info("domain:%s" % (domain))
            print_data_distribution(train_data, val_data, test_data)

        if args.algorithm == "feddf":
            server_domain = "mnist"
            trainset = ImageFolder("data/digit/" + server_domain + "/train_images", train_transform)
            size = len(trainset)
            index = np.arange(size)
            np.random.shuffle(index)
            self.server_dataset = Subset(trainset, index[:32000])
            return self.trainsets, self.valsets, self.testsets, self.server_dataset
        else:
            return self.trainsets, self.valsets, self.testsets


class Office(object):

    def __init__(self):
        pass

    def data_partition(self):
        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainsets = []
        self.valsets = []
        self.testsets = []
        for domain in ["amazon", "dslr", "webcam", "caltech"]:
            dataset = ImageFolder("data/office/" + domain)
            train_data, valtest_data = train_test_split(dataset, test_size=0.5, random_state=1)
            val_data, test_data = train_test_split(valtest_data, test_size=0.5, random_state=1)
            self.trainsets.append(CustomDataset(train_data, train_transform))
            self.valsets.append(CustomDataset(val_data, test_transform))
            self.testsets.append(CustomDataset(test_data, test_transform))

            logger.info("domain:%s" % (domain))
            print_data_distribution(train_data, val_data, test_data)

        return self.trainsets, self.valsets, self.testsets


class Domainnet(object):

    def __init__(self):
        pass

    def data_partition(self):
        train_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainsets = []
        self.valsets = []
        self.testsets = []
        for domain in ["clipart", "infograph", "quickdraw", "real", "sketch"]:
            dataset = ImageFolder("data/domainnet-4/" + domain)
            train_data, valtest_data = train_test_split(dataset, test_size=0.5, random_state=1)
            val_data, test_data = train_test_split(valtest_data, test_size=0.5, random_state=1)
            self.trainsets.append(CustomDataset(train_data, train_transform))
            self.valsets.append(CustomDataset(val_data, test_transform))
            self.testsets.append(CustomDataset(test_data, test_transform))

            logger.info("domain:%s" % (domain))
            print_data_distribution(train_data, val_data, test_data)

        return self.trainsets, self.valsets, self.testsets


if __name__ == "__main__":
    fmnist = FMNIST()
    fmnist.data_partition()
