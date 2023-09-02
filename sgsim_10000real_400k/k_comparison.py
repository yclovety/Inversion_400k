import  time
from random import random
import pyro
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim  import DCTAdam
from BNN.sgsim_10000real_400k.node import single_qwt

x_data=np.zeros((1,400))
for root, dirs, files in os.walk('sgsimdata'):
    for file in files:
        # print(os.path.join(root, file))
        new =np.loadtxt(os.path.join(root, file)).T
        x_data=np.concatenate((x_data,new))
x_data=np.delete(x_data,0,axis=0)
scaler_k=StandardScaler()
normalized_k=scaler_k.fit_transform(x_data)
y_data=np.loadtxt("qwt_nonoise_change_sgsim10000.txt")
m=10
n=4000
y_data=(y_data*1e5).reshape((10000,n))[:,::m]####切片操作隔M列
scaler_q=StandardScaler()
normalized_q=scaler_q.fit_transform(y_data)

k_testtensor=torch.from_numpy((normalized_k[9000:10000,:]))
q_testtensor=torch.from_numpy((normalized_q[9000:10000, :]))
test_dataset=torch.utils.data.TensorDataset(q_testtensor, k_testtensor)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=1)
#
input_size=400
hidden_size=800
output_size=400
# 定义贝叶斯神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super( SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out =self.fc2(out)
        return out
SimpleNN = SimpleNN(input_size, hidden_size, output_size)


sigma=0.1
def model(x, y, input_size, hidden_size, output_size):
    # 定义神经网络的权重和偏置的先验分布
    fc1_weight_prior = dist.Normal(
        torch.zeros(hidden_size, input_size), torch.ones(hidden_size, input_size)*sigma
    ).to_event(2)
    fc1_bias_prior = dist.Normal(torch.zeros(hidden_size), torch.ones(hidden_size)*sigma).to_event(1)
    fc2_weight_prior = dist.Normal(
        torch.zeros(output_size, hidden_size), torch.ones(output_size, hidden_size)*sigma
    ).to_event(2)
    fc2_bias_prior = dist.Normal(torch.zeros(output_size), torch.ones(output_size)*sigma).to_event(1)

    priors = {
        "fc1.weight": fc1_weight_prior,
        "fc1.bias": fc1_bias_prior,
        "fc2.weight": fc2_weight_prior,
        "fc2.bias": fc2_bias_prior,
    }

    # 将神经网络变为贝叶斯神经网络
    lifted_module = pyro.random_module("module", SimpleNN, priors)

    lifted_nn = lifted_module()

    with pyro.plate("data", x.shape[0]):
        prediction_mean = lifted_nn(x)
        # 使用多元高斯分布建模输出
        pyro.sample("obs", dist.MultivariateNormal(prediction_mean, torch.eye(output_size)), obs=y)

def guide(x, y, input_size, hidden_size, output_size):
    # 为神经网络的权重和偏置定义变分分布
    fc1_weight_loc = pyro.param("fc1_weight_loc", torch.randn(hidden_size, input_size))
    fc1_weight_scale = pyro.param("fc1_weight_scale", torch.ones(hidden_size, input_size)*sigma, constraint=dist.constraints.positive)
    fc1_bias_loc = pyro.param("fc1_bias_loc", torch.randn(hidden_size))
    fc1_bias_scale = pyro.param("fc1_bias_scale", torch.ones(hidden_size)*sigma, constraint=dist.constraints.positive)

    fc2_weight_loc = pyro.param("fc2_weight_loc", torch.randn(output_size, hidden_size))
    fc2_weight_scale = pyro.param("fc2_weight_scale", torch.ones(output_size, hidden_size)*sigma, constraint=dist.constraints.positive)
    fc2_bias_loc = pyro.param("fc2_bias_loc", torch.randn(output_size))
    fc2_bias_scale = pyro.param("fc2_bias_scale", torch.ones(output_size)*sigma, constraint=dist.constraints.positive)

    variational_dist = {
        "fc1.weight": dist.Normal(fc1_weight_loc, fc1_weight_scale).to_event(2),
        "fc1.bias": dist.Normal(fc1_bias_loc, fc1_bias_scale).to_event(1),
        "fc2.weight": dist.Normal(fc2_weight_loc, fc2_weight_scale).to_event(2),
        "fc2.bias": dist.Normal(fc2_bias_loc, fc2_bias_scale).to_event(1),
    }

    lifted_module = pyro.random_module("module", SimpleNN, variational_dist)

    return lifted_module()

pyro.get_param_store().load('bnn_model_params0.1.pt')

def predict(x, num_samples=200):
    predictions = []
    for _ in range(num_samples):
        sampled_nn = guide(None, None, input_size, hidden_size, output_size)
        pred = sampled_nn(x)
        predictions.append(pred)
    return torch.stack(predictions)


##指标1
# start_time = time.time()
# percent=np.zeros(len(k_testtensor))
# MSE=np.zeros(len(k_testtensor))
# predict_kall=[]
# for i in range(len(q_testtensor)):
#     output=predict(q_testtensor[i].float()).detach().numpy()
#     predict_k=scaler_k.inverse_transform(output.reshape(1,400)).reshape(400,)
#     predict_kall.append(predict_k)
#     print('pred%04d:' % i )
#     print(predict_k)
#     ture_k=x_data[9000+i,:]
#     percent[i] = (abs(predict_k - ture_k) / ture_k).mean()
#     MSE[i] = (abs(predict_k - ture_k) ** 2).mean()
# end_time = time.time()
# execution_time = end_time - start_time
# predict_kall=np.array(predict_kall)
# np.savetxt('predict_k2',predict_kall)
# print("代码块执行时间：", execution_time, "秒")
#
# fig,ax =plt.subplots(2,1,figsize=(8,6))
# labels=("0-10%","10-20%","20-30%","30-40",">40%")
# a=np.zeros(5)
# for i in range(len(k_testtensor)):
#     if percent[i]<0.1:
#         a[0]+=1
#     elif percent[i]<0.2:
#         a[1]+=1
#     elif percent[i]<0.3:
#         a[2]+=1
#     elif percent[i]<0.4:
#         a[3]+=1
#     else:
#         a[4]+=1
#
# ax[0].pie(a/len(k_testtensor),labels=labels,autopct="%1.1f%%")
# ax[0].axis=('equal')
# ax[0].set_title('average Relative Square Error ')
# y=range(len(k_testtensor))
# ax[1].bar(y,MSE)
# ax[1].set_xlabel("test_number")
# ax[1].set_ylabel("MSE")
# plt.show()


#指标2   三维图像
# output=predict(q_testtensor[0].float()).detach().numpy()
# predict_k=scaler_k.inverse_transform(output)
# bins=np.arange(0,15)
# histall=[]
# for i in range(400):
#     hist, _ = np.histogram(predict_k[:,i], bins=bins)
#     histall.append(hist/200)
#
# # 创建三维图形对象
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# # 绘制三维散点图
# for z in range(3):
#     k=bins[:14]
#     density=np.array(histall[z])
#     # color=plt.cm.plasma(random.choice(range(plt.cm.plasma.N)))
#     ax.bar(k,density,zs=z,zdir='x')
# ax.set_xlabel('k_number')
# ax.set_ylabel('k_value')
# ax.set_zlabel('probability')
# plt.show()

#####检测是否是求均值导致qwt与原始的不同
test_num=1
output=predict(q_testtensor[test_num].float()).detach().numpy()
predict_k=scaler_k.inverse_transform(output)
predict_qwtall=np.zeros((predict_k.shape[0],400))
ture_qwt=y_data[9000+test_num,:]
percent=np.zeros(predict_k.shape[0])
MSE=np.zeros(predict_k.shape[0])
for i in range(predict_k.shape[0]):
    for j in range(400):
        predict_k[i,j]=2**predict_k[i,j]*1e-16
    predict_qwt=single_qwt(predict_k[i,:])*1e5
    predict_qwt=np.array(predict_qwt)[::m]
    predict_qwtall[i,:]=predict_qwt
    percent[i] = (abs(predict_qwt - ture_qwt) / ture_qwt).mean()
    MSE[i] = (abs(predict_qwt - ture_qwt) ** 2).mean()

fig,ax =plt.subplots(2,1,figsize=(8,6))
labels=("0-10%","10-20%","20-30%","30-40",">40%")
a=np.zeros(5)
for i in range(predict_k.shape[0]):
    if percent[i]<0.1:
        a[0]+=1
    elif percent[i]<0.2:
        a[1]+=1
    elif percent[i]<0.3:
        a[2]+=1
    elif percent[i]<0.4:
        a[3]+=1
    else:
        a[4]+=1
ax[0].pie(a/predict_k.shape[0],labels=labels,autopct="%1.1f%%")
ax[0].axis=('equal')
ax[0].set_title('average Relative Square Error of qwt( q_k_q )')
y=range(predict_k.shape[0])
ax[1].bar(y,MSE)
ax[1].set_xlabel("test_number")
ax[1].set_ylabel("MSE")
plt.show()
np.savetxt('predict_qwt1',predict_qwtall)