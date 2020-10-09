import torch
import torch.nn as nn
# from model_res import ResNet18
from model2_batch import VGG
import torch.optim as optim
import numpy as np
# import torch.utils.data as data
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import os
from pgd_attack import  attack

device='cuda'
# net=ResNet18()
net=VGG()
net.to(device)

torch.set_num_threads(4)
#########  finetune joint
# ckpt=torch.load('./checkpoint/ckpt_40.tar')
# net.load_state_dict(ckpt['model_state_dict'])


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# learning_rate=1e-3 #finetune 1e-4 \ ori 1e-3
optimizer=optim.Adam(net.parameters())
# optimizer.load_state_dict(ckpt['optimizer_state_dict'])

loss_cls=nn.CrossEntropyLoss()
# loss_reg=nn.

num_epochs=100

def adjust_learning_rate(optimizer,epoch,learning_rate):

    # lr=learning_rate/(10**(epoch//30))
    # # lr=learning_rate
    # if epoch>=15:
    #     lr=learning_rate/10
    # else:
    lr=learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr']=lr



def train( ):

    # count=0
    for epoch in range(num_epochs):

        # gamma=10*epoch/num_epochs

        train_loss=0
        test_loss=0
        total=0
        total_test=0
        correct=0
        correct_test=0


        net.train()
        # adjust_learning_rate(optimizer, epoch, learning_rate)
        print('epoch',epoch)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # inputs=attack(inputs,targets,net,loss_cls)


            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_cls(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # print('loss',loss.item())


        # if (epoch+1)%10==0:
        ckpt_name='./checkpoint_weight/ckpt_'+str(epoch)+'.tar'
        print('Saving..')
        if not os.path.isdir('checkpoint_weight'):
            os.mkdir('checkpoint_weight')

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss / total
        }, ckpt_name)



        net.eval()
        # with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_cls(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct_test / total_test, correct_test, total_test))

if __name__=='__main__':
    train()