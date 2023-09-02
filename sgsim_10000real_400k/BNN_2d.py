# Bayesian Convolutional Neural Network（BCNN）的训练代码与传统的卷积神经网络类似，但需要使用贝叶斯方法来对权重进行采样和后验分布的计算。下面是一个简单的BCNN训练代码示例，使用PyTorch框架实现：
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.distributions.normal import Normal


class BCNN(nn.Module):
    def __init__(self,num_qwt):
        super(BCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(10,20, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, num_qwt)
        self.B1= nn.BatchNorm2d(10)
        self.B2 = nn.BatchNorm2d(20)
        # 定义权重的先验分布
        self.prior_sigma1 = 1.0
        self.prior_sigma2 = 1.0

        # 定义权重的后验分布
        self.posterior_mu1 = nn.Parameter(torch.randn_like(self.conv1.weight))
        self.posterior_mu2 = nn.Parameter(torch.randn_like(self.conv2.weight))
        self.posterior_mu3 = nn.Parameter(torch.randn_like(self.fc1.weight))
        self.posterior_mu4 = nn.Parameter(torch.randn_like(self.fc2.weight))
        self.posterior_rho1 = nn.Parameter(torch.randn_like(self.conv1.weight))
        self.posterior_rho2 = nn.Parameter(torch.randn_like(self.conv2.weight))
        self.posterior_rho3 = nn.Parameter(torch.randn_like(self.fc1.weight))
        self.posterior_rho4 = nn.Parameter(torch.randn_like(self.fc2.weight))

    def forward(self, x):
        # 采样权重
        sigma1 = torch.log(1 + torch.exp(self.posterior_rho1))
        sigma2 = torch.log(1 + torch.exp(self.posterior_rho2))
        sigma3 = torch.log(1 + torch.exp(self.posterior_rho3))
        sigma4 = torch.log(1 + torch.exp(self.posterior_rho4))
        epsilon1 = Normal(0, 1).sample(self.conv1.weight.shape).to(x.device)
        epsilon2 = Normal(0, 1).sample(self.conv2.weight.shape).to(x.device)
        epsilon3 = Normal(0, 1).sample(self.fc1.weight.shape).to(x.device)
        epsilon4 = Normal(0, 1).sample(self.fc2.weight.shape).to(x.device)
        w1 = self.posterior_mu1 + sigma1 * epsilon1
        w2 = self.posterior_mu2 + sigma2 * epsilon2
        w3 = self.posterior_mu3 + sigma3 * epsilon3
        w4 = self.posterior_mu4 + sigma4 * epsilon4

        # 前向传播
        x =F.conv2d(x, w1, padding=0)
        x= self.B1(x)
        x= F.relu(x)
        x = F.max_pool2d(x, 2)
        x =F.conv2d(x, w2, padding=1)
        x= self.B2(x)
        x=F.relu(x)
        x = F.max_pool2d(x, (2,1))
        x =x.reshape(x.size(0), -1)
        x = F.relu(F.linear(x, w3))
        x = F.dropout(x, training=self.training)
        x = F.linear(x, w4)
        return x

    def log_prior(self):
        # 计算权重的先验分布概率
        return (-self.conv1.weight.pow(2).sum() / (2 * self.prior_sigma1 ** 2)
                -self.conv2.weight.pow(2).sum() / (2 * self.prior_sigma1 ** 2)
                -self.fc1.weight.pow(2).sum() / (2 * self.prior_sigma2 ** 2)
                -self.fc2.weight.pow(2).sum() / (2 * self.prior_sigma2 ** 2))

    def log_posterior(self):
        # 计算权重的后验分布概率
        return (-((self.conv1.weight - self.posterior_mu1).pow(2) / (2 * torch.log(1 + torch.exp(self.posterior_rho1)).pow(2))).sum()
                -((self.conv2.weight - self.posterior_mu2).pow(2) / (2 * torch.log(1 + torch.exp(self.posterior_rho2)).pow(2))).sum()
                -((self.fc1.weight - self.posterior_mu3).pow(2) / (2 * torch.log(1 + torch.exp(self.posterior_rho3)).pow(2))).sum()
                -((self.fc2.weight - self.posterior_mu4).pow(2) / (2 * torch.log(1 + torch.exp(self.posterior_rho4)).pow(2))).sum())

    def kl_divergence(self):
        # 计算KL散度
        return self.log_posterior() - self.log_prior()

# 加载数据集
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_data=np.zeros((1,400))
for root, dirs, files in os.walk('sgsimdata'):
    for file in files:
        # print(os.path.join(root, file))
        new =np.loadtxt(os.path.join(root, file)).T
        x_data=np.concatenate((x_data,new))

x_data=np.delete(x_data,0,axis=0)

y_data=np.loadtxt("qwt_nonoise_change_sgsim10000.txt")
m=20
n=4000
y_data=(y_data*1e5).reshape((10000,n))[:,2::m]####切片操作隔15列

# min_val = np.min(y_data)
# max_val = np.max(y_data)

x_traintensor=torch.from_numpy(x_data[:9000,:]).reshape(9000,1,20,20)
y_traintensor=torch.from_numpy(y_data[:9000,:])
nreal=len(x_traintensor)
x_testtensor=torch.from_numpy((x_data[9000:10000,:])).reshape(1000,1,20,20)
y_testtensor=torch.from_numpy((y_data[9000:10000,:]))

train_dataset=torch.utils.data.TensorDataset(x_traintensor,y_traintensor)
test_dataset=torch.utils.data.TensorDataset(x_testtensor,y_testtensor)
# Hyper parameters
num_epochs =50
batch_size =100
num_qwt=200
learning_rate = 0.001
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=1)


# 定义模型
model = BCNN(num_qwt).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# 训练模型
num_epochs = 10
train_lossall=np.zeros(num_epochs)
test_lossall=np.zeros(num_epochs)
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device).to(torch.float32)
        labels = labels.to(device).to(torch.float32)
        outputs = model(images)
        loss =  criterion(outputs, labels) + 1e-5 * model.kl_divergence()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
    train_lossall[epoch]=train_loss/len(train_loader)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32)
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            # _, predicted = torch.max(outputs.data, 1)
    test_lossall[epoch] =test_loss/ len(test_loader)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
          .format(epoch + 1, num_epochs, train_lossall[epoch], test_lossall[epoch]))
    torch.save(model.state_dict(), 'qwtpredict_10000_200_change_BNN.ckpt')


print(train_lossall)
print(test_lossall)

plt.plot(range(num_epochs),train_lossall)
plt.plot(range(num_epochs),test_lossall)
plt.show()

# 在上面的代码中，我们定义了一个BCNN模型，并使用PyTorch自带的MNIST数据集进行训练。在定义模型时，我们将权重的先验分布设置为标准正态分布，将后验分布设置为模型参数，并使用采样方法来计算权重的后验分布。在训练过程中，我们不仅需要最小化交叉熵损失，还需要最小化KL散度，从而训练出一个具有较好泛化性能的网络模型。