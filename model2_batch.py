'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AB(nn.Module):
    def __init__(self,b,h,w):
        super(AB, self).__init__()
        self.A=torch.nn.Parameter(torch.ones(b,1,h,w))
        self.B=torch.nn.Parameter(torch.zeros(b,1,h,w))

    def forward(self, x):
        out=self.A*x+self.B
        return out


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()



        # self.features = self._make_layers(cfg[vgg_name])
        self.relu=nn.ReLU(inplace=True)

        self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(64)

        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn4=nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.bn5=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn6=nn.BatchNorm2d(256)
        self.conv7=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn7=nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.bn8=nn.BatchNorm2d(512)
        self.conv9=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn9=nn.BatchNorm2d(512)
        self.conv10=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn10=nn.BatchNorm2d(512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn11=nn.BatchNorm2d(512)
        self.conv12=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn12=nn.BatchNorm2d(512)
        self.conv13=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn13=nn.BatchNorm2d(512)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, 10)

    def zzz(self,Conv,x,sample_num):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        x1 = Conv(x)
        # x1=A*x1+B###b,c,h,w
        ABnet = AB(b, h, w).cuda()
        optimizer1 = optim.RMSprop(ABnet.parameters(), lr=1e-3)
        ########label?
        x_unfold = nn.Unfold(kernel_size=3, padding=1)(x).view(b, -1, h * w)  # b,c*9,h*w

        # del x_unfold,x_diff,x_diff_cube
        gamma = 1000
        for iter in range(10):
            optimizer1.zero_grad()

            sample_ind = torch.randint(0, h * w, (b, 3 * sample_num))

            x1_ = ABnet(x1)  # b,c1,h,w
            x1_ = x1_.view(b, -1, h * w)  # b,c1,h*w
            x1_batch = torch.zeros(b, x1_.size(1), 3 * sample_num).cuda()
            x_batch = torch.zeros(b, x_unfold.size(1), 3 * sample_num).cuda()
            for i_b in range(b):
                x1_batch[i_b, :, :] = x1_[i_b, :, sample_ind[i_b, :]]  ###  b,c1,3*256
                x_batch[i_b, :, :] = x_unfold[i_b, :, sample_ind[i_b, :]]  ###b,c*9,3*256

            x1_batch = x1_batch.view(b, -1, sample_num, 3).permute(0, 2, 1, 3).contiguous().view(b * sample_num, -1, 3)
            x_batch = x_batch.view(b, -1, sample_num, 3).permute(0, 2, 1, 3).contiguous().view(b * sample_num, -1, 3)

            x1_diff1 = torch.mean((x1_batch[:, :, 0] - x1_batch[:, :, 1]) ** 2, dim=1)
            x1_diff2 = torch.mean((x1_batch[:, :, 0] - x1_batch[:, :, 2]) ** 2, dim=1)
            x1_diff_diff = x1_diff1 - x1_diff2  ###b*256

            x_diff1 = torch.mean((x_batch[:, :, 0] - x_batch[:, :, 1]) ** 2, dim=1)
            x_diff2 = torch.mean((x_batch[:, :, 0] - x_batch[:, :, 2]) ** 2, dim=1)
            x_diff_diff = x_diff1 - x_diff2  ###b*256
            sign_diff_diff = torch.sign(x_diff_diff)

            unary = torch.sum(torch.max(torch.zeros_like(x1_diff_diff), -1 * sign_diff_diff * x1_diff_diff)) / b * sample_num
            regular = (torch.sum((ABnet.A - torch.ones_like(ABnet.A)) ** 2) + torch.sum(ABnet.B ** 2)) / (b * h * w)
            # print(iter,unary)
            # print(iter,gamma*regular)
            loss_triplet = unary + gamma * regular

            loss_triplet.backward(retain_graph=True)  ###
            optimizer1.step()
            # print(iter)
        # del x_unfold,x
        x1 = ABnet(x1)
        return x1


    def forward(self, x):


        x=self.relu(self.bn1(self.zzz(self.conv1,x,256)))


####################

        x=self.relu(self.bn2(self.conv2(x)))

        x=self.maxpool1(x)


        x=self.relu(self.bn3(self.zzz(self.conv3,x,128)))

        x=self.relu(self.bn4(self.conv4(x)))

        x=self.maxpool2(x)

        # x=self.dynamic5(x,self.conv5)
        x=self.relu(self.bn5(self.zzz(self.conv5,x,64)))
        # x=self.dynamic6(x,self.conv6)
        x=self.relu(self.bn6(self.conv6(x)))
        # x=self.dynamic7(x,self.conv7)
        x=self.relu(self.bn7(self.conv7(x)))

        x=self.maxpool3(x)

        # x=self.dynamic8(x,self.conv8)
        x=self.relu(self.bn8(self.zzz(self.conv8,x,64)))
        # x=self.dynamic9(x,self.conv9)
        x=self.relu(self.bn9(self.conv9(x)))
        # x=self.dynamic10(x,self.conv10)
        x=self.relu(self.bn10(self.conv10(x)))

        x=self.maxpool4(x)

        # x=self.dynamic11(x,self.conv11)
        x=self.relu(self.bn11(self.conv11(x)))
        # x=self.dynamic12(x,self.conv12)
        x=self.relu(self.bn12(self.conv12(x)))
        # x=self.dynamic13(x,self.conv13)
        x=self.relu(self.bn13(self.conv13(x)))

        x=self.maxpool5(x)

        x = x.view(-1,512)
        x = self.classifier(x)
        return x




def test():
    net = VGG()
    device='cuda'
    net.to(device)
    x = torch.randn(128,3,32,32).to(device)
    y = net(x)
    print(y)

# test()
if __name__=='__main__':
    test()
