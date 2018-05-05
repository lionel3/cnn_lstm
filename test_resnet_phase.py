import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os
from PIL import Image

import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda
import copy

parser = argparse.ArgumentParser(description='cnn_lstm testing')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--test', default=800, type=int, help='test batch size, default 800')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')
parser.add_argument('-n', '--name', type=str, help='name of model')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
test_batch_size = args.test
model_name = args.name
workers = args.work
crop_type = args.crop

model_pure_name, _ = os.path.splitext(model_name)
pred_name = model_pure_name + '_pred_crop_' + str(crop_type) + '.pkl'

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))

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
class pure_resnet(torch.nn.Module):
    def __init__(self):
        super(pure_resnet, self).__init__()
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
        self.fc1 = nn.Linear(2048, 7)
        init.xavier_uniform(self.fc1.weight)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        y = self.fc1(x)
        return y

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

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each

def test_model(test_dataset, test_num_each):
    num_test = len(test_dataset)
    test_idx = [i for i in range(num_test)]
    print('num of test dataset: {:6d}'.format(num_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        num_workers=workers,
        pin_memory=False
    )
    model = pure_resnet()
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_name))
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    test_start_time = time.time()
    for data in test_loader:
        inputs, labels_1, labels_2 = data
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels_2, volatile=True)

        if crop_type == 0 or crop_type == 1:
            outputs = model.forward(inputs)
        elif crop_type == 5:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            outputs = model.forward(inputs)
            outputs = outputs.view(5, -1, 7)
            outputs = torch.mean(outputs, 0)
        elif crop_type == 10:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            inputs = inputs.view(-1, 3, 224, 224)
            outputs = model.forward(inputs)
            outputs = outputs.view(10, -1, 7)
            outputs = torch.mean(outputs, 0)
        _, preds = torch.max(outputs.data, 1)
        for i in range(len(preds)):
            all_preds.append(preds[i])
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        test_corrects += torch.sum(preds == labels.data)
        # print(test_corrects)
    test_elapsed_time = time.time() - test_start_time
    test_accuracy = test_corrects / num_test
    test_average_loss = test_loss / num_test


    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = model_pure_name + '_test_' + str(save_test)+'_crop_' + str(crop_type) + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)
    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test loss: {:4.4f}'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss, test_accuracy))

print()


def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')
    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
