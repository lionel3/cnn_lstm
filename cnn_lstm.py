import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import DataParallel
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
import copy
from torchvision.transforms import Lambda

parser = argparse.ArgumentParser(description='lstm Training')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=100, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=8, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=1, type=int, help='num of workers to use, default 1')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        # self.target_transform=target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)


class multi_lstm(torch.nn.Module):
    def __init__(self):
        super(multi_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.fc2 = nn.Linear(2048, 7)
        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        init.xavier_uniform(self.fc.weight)
        init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fc(y)
        return z, y

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, val_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)

    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    #num_train_we_use = 8000
    #num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    #    np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of valid dataset: {:6d}'.format(num_val))
    print('num of valid we use : {:6d}'.format(num_val_we_use))
    print('num of all valid use: {:6d}'.format(num_val_all))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_idx,
        num_workers=workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_idx,
        num_workers=workers,
        pin_memory=False
    )
    model = multi_lstm()
    if use_gpu:
        model = model.cuda()

    model = DataParallel(model)
    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)
    criterion_2 = nn.CrossEntropyLoss(size_average=False)
    sig_f = nn.Sigmoid()

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters())
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': 1e-3},
                {'params': model.module.fc.parameters(), 'lr': 1e-3},
            ], lr=1e-4, momentum=0.9)

            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': 1e-3},
                {'params': model.module.fc.parameters(), 'lr': 1e-3},
            ], lr=1e-4)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy_1 = 0.0
    best_val_accuracy_2 = 0.0  # judge by accu2
    correspond_train_acc_1 = 0.0
    correspond_train_acc_2 = 0.0

    all_info = []
    all_train_accuracy_1 = []
    all_train_accuracy_2 = []
    all_train_loss_1 = []
    all_train_loss_2 = []
    all_val_accuracy_1 = []
    all_val_accuracy_2 = []
    all_val_loss_1 = []
    all_val_loss_2 = []

    for epoch in range(epochs):
        # np.random.seed(epoch)
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_idx,
            num_workers=args.work,
            pin_memory=False
        )

        model.train()
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        train_corrects_1 = 0
        train_corrects_2 = 0

        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels_1 = Variable(labels_1.cuda())
                labels_2 = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels_1 = Variable(labels_1)
                labels_2 = Variable(labels_2)

            optimizer.zero_grad()

            outputs_1, outputs_2 = model.forward(inputs)

            _, preds_2 = torch.max(outputs_2.data, 1)

            sig_out = outputs_1.data.cpu()
            sig_out = sig_f(sig_out)
            preds_1 = torch.ByteTensor(sig_out > 0.5)
            preds_1 = preds_1.long()
            train_corrects_1 += torch.sum(preds_1 == labels_1.data.cpu())
            labels_1 = Variable(labels_1.data.float())

            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()

            train_loss_1 += loss_1.data[0]
            train_loss_2 += loss_2.data[0]
            train_corrects_2 += torch.sum(preds_2 == labels_2.data)
        train_elapsed_time = time.time() - train_start_time
        train_accuracy_1 = train_corrects_1 / num_train_all / 7
        train_accuracy_2 = train_corrects_2 / num_train_all
        train_average_loss_1 = train_loss_1 / num_train_all / 7
        train_average_loss_2 = train_loss_2 / num_train_all

        # begin eval

        model.eval()
        val_loss_1 = 0.0
        val_loss_2 = 0.0
        val_corrects_1 = 0
        val_corrects_2 = 0

        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels_1 = Variable(labels_1.cuda())
                labels_2 = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels_1 = Variable(labels_1)
                labels_2 = Variable(labels_2)

            outputs_1, outputs_2 = model.forward(inputs)

            _, preds_2 = torch.max(outputs_2.data, 1)

            sig_out = outputs_1.data.cpu()
            sig_out = sig_f(sig_out)
            preds_1 = torch.ByteTensor(sig_out > 0.5)
            preds_1 = preds_1.long()
            val_corrects_1 += torch.sum(preds_1 == labels_1.data.cpu())
            labels_1 = Variable(labels_1.data.float())

            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            val_loss_1 += loss_1.data[0]
            val_loss_2 += loss_2.data[0]
            val_corrects_2 += torch.sum(preds_2 == labels_2.data)
        val_elapsed_time = time.time() - val_start_time
        val_accuracy_1 = val_corrects_1 / num_val_all / 7
        val_accuracy_2 = val_corrects_2 / num_val_all
        val_average_loss_1 = val_loss_1 / num_val_all / 7
        val_average_loss_2 = val_loss_2 / num_val_all
        print('epoch: {:4d} '
              'train time: {:2.0f}m{:2.0f}s'
              ' train loss_1: {:4.4f}'
              ' train accu_1: {:.4f}'
              ' valid time: {:2.0f}m{:2.0f}s'
              ' valid loss_1: {:4.4f}'
              ' valid accu_1: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_1,
                      train_accuracy_1,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss_1,
                      val_accuracy_1))
        print('epoch: {:4d}'
              ' train time: {:2.0f}m{:2.0f}s'
              ' train loss_2: {:4.4f}'
              ' train accu_2: {:.4f}'
              ' valid time: {:2.0f}m{:2.0f}s'
              ' valid loss_2: {:4.4f}'
              ' valid accu_2: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_2,
                      train_accuracy_2,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss_2,
                      val_accuracy_2))

        if optimizer_choice == 0:
            exp_lr_scheduler.step(val_average_loss_1 + val_average_loss_2)

        if val_accuracy_2 > best_val_accuracy_2 and val_accuracy_1 > 0.95:
            best_val_accuracy_2 = val_accuracy_2
            best_val_accuracy_1 = val_accuracy_1
            correspond_train_acc_1 = train_accuracy_1
            correspond_train_acc_2 = train_accuracy_2
            best_model_wts = copy.deepcopy(model.state_dict())
        elif val_accuracy_2 == best_val_accuracy_2 and val_accuracy_1 > 0.95:
            if val_accuracy_1 > best_val_accuracy_1:
                correspond_train_acc_1 = train_accuracy_1
                correspond_train_acc_2 = train_accuracy_2
                best_model_wts = copy.deepcopy(model.state_dict())
            elif val_accuracy_1 == best_val_accuracy_1:
                if train_accuracy_2 > correspond_train_acc_2:
                    correspond_train_acc_2 = train_accuracy_2
                    correspond_train_acc_1 = train_accuracy_1
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif train_accuracy_2 == correspond_train_acc_2:
                    if train_accuracy_1 > best_val_accuracy_1:
                        correspond_train_acc_1 = train_accuracy_1
                        best_model_wts = copy.deepcopy(model.state_dict())

        all_train_loss_1.append(train_average_loss_1)
        all_train_loss_2.append(train_average_loss_2)
        all_train_accuracy_1.append(train_accuracy_1)
        all_train_accuracy_2.append(train_accuracy_2)
        all_val_loss_1.append(val_average_loss_1)
        all_val_loss_2.append(val_average_loss_2)
        all_val_accuracy_1.append(val_accuracy_1)
        all_val_accuracy_2.append(val_accuracy_2)

    all_info.append(all_train_accuracy_1)
    all_info.append(all_train_accuracy_2)
    all_info.append(all_train_loss_1)
    all_info.append(all_train_loss_2)
    all_info.append(all_val_accuracy_1)
    all_info.append(all_val_accuracy_2)
    all_info.append(all_val_loss_1)
    all_info.append(all_val_loss_2)

    # model.load_state_dict(best_model_wts)
    print('best accuracy_1: {:.4f} cor train accu_1: {:.4f}'.format(best_val_accuracy_1, correspond_train_acc_1))
    print('best accuracy_2: {:.4f} cor train accu_2: {:.4f}'.format(best_val_accuracy_2, correspond_train_acc_2))
    save_val_1 = int("{:4.0f}".format(best_val_accuracy_1 * 10000))
    save_val_2 = int("{:4.0f}".format(best_val_accuracy_2 * 10000))
    save_train_1 = int("{:4.0f}".format(correspond_train_acc_1 * 10000))
    save_train_2 = int("{:4.0f}".format(correspond_train_acc_2 * 10000))
    model_name = "cnn_lstm" \
                 + "_epoch_" + str(epochs) \
                 + "_length_" + str(sequence_length) \
                 + "_opt_" + str(optimizer_choice) \
                 + "_mulopt_" + str(multi_optim) \
                 + "_batch_" + str(train_batch_size) \
                 + "_train1_" + str(save_train_1) \
                 + "_train2_" + str(save_train_2) \
                 + "_val1_" + str(save_val_1) \
                 + "_val2_" + str(save_val_2)\
                 + ".pth"

    torch.save(best_model_wts, model_name)

    record_name = "cnn_lstm" \
                  + "_epoch_" + str(epochs) \
                  + "_length_" + str(sequence_length) \
                  + "_opt_" + str(optimizer_choice) \
                  + "_mulopt_" + str(multi_optim) \
                  + "_batch_" + str(train_batch_size) \
                  + "_train1_" + str(save_train_1) \
                  + "_train2_" + str(save_train_2) \
                  + "_val1_" + str(save_val_1) \
                  + "_val2_" + str(save_val_2) \
                  + ".pkl"

    with open(record_name, 'wb') as f:
        pickle.dump(all_info, f)
    print()


def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
