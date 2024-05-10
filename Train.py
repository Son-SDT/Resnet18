import Dataset as ds
import Architecture as arch
import torch,torchaudio, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch.autograd import Variable
from torch import nn
import os

current_path = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_path,"Dataset")

transform = Compose([
    ToTensor(),
    Resize((224,224)),
])
#train_set = CIFAR10('./data', train=True, transform=data_tf, download=True)
train_set = ds.AnimalDataset(root=data_root,train=True, transform = transform)
train_data = DataLoader(train_set, batch_size = 64, shuffle=True)
#test_set = CIFAR10('./data', train=False, transform=data_tf, download=True)
test_set = ds.AnimalDataset(root=data_root,train=False, transform = transform)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)

net = arch.Resnet18(3, 10)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

from datetime import datetime


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#from torch.nn.parallel import DistributedDataParallel
#net = nn.DataParallel(net)

def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        print('cuda is available')
        net = net.to(device)
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.to(device))
                
                label = Variable(label.to(device))
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.to(device), volatile=True)
                   
                    label = Variable(label.to(device), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        #
        
        prev_time = cur_time
        print(epoch_str + time_str)
    print('Saving...')
    result_path = os.join(current_path,'Result')
    name = f'resnet34{len(os.listdir(result_path)+1)}.pt'
    torch.save(net.state_dict(),os.path.join(result_path,name))   
    print(f'Saved {name}')

train(net, train_data, test_data, 5, optimizer, criterion)
