#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

class Linear_net(nn.Module):
    def __init__(self, feature_dim, classes_num):
        super(Linear_net, self).__init__()
        self.classes_num = classes_num
        self.feature_dim = feature_dim
        self.model = nn.Sequential(
                nn.Linear(feature_dim, 20)
                nn.Linear(20, 20)
                nn.Linear(20, self.classes_num)
            )

    def forward(self, x):
        x = self.model(x)
        return x


# read numpy data and convert it to pytotch tensor
class tx_dataset(Dataset):

    def __init__(self, data_path, label_path):
        data_tensor = torch.from_numpy(np.load(data_path))
        target_tensor = torch.from_numpy(np.load(label_path))
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def train_batch(model, optimizer, criterion, batch, label): 
    label_var = Variable(label)
    output = model(Variable(batch))
    # compute gradient and do SGD step
    optimizer.zero_grad() # 
    criterion(output, label_var).backward()
    optimizer.step()

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, criterion, batch, label)
        if num_batches%50 == 0:
            temp_str = '%23s%-9s%-13s'%(('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
            print(temp_str)
        num_batches +=1


num_batches = 0
data_path = ''
label_path = ''

tensor_dataset = tx_dataset(data_path, label_path)

tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=32,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

def main():
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    model = Linear_net(feature_dim=5, classes_num=7)
    optimizer = optim.SGD(model.parameters(), lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    print("start training!")
    for epoch in range(epochs):
        print("the {}th epoch: ".format(epoch))
        train_epoch(model, tensor_dataloader, criterion, optimizer)


if __name__ == "__main__":
    main()


