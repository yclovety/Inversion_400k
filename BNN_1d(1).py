import os
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim  import DCTAdam
#

x_data=np.zeros((1,400))
for root, dirs, files in os.walk('sgsimdata'):
    for file in files:
        # print(os.path.join(root, file))
        new =np.loadtxt(os.path.join(root, file)).T
        x_data=np.concatenate((x_data,new))
x_data=np.delete(x_data,0,axis=0)
scaler_k=MinMaxScaler()
normalized_k=scaler_k.fit_transform(x_data)

y_data=np.loadtxt("qwt_nonoise_change_sgsim10000.txt")
m=10
n=4000
y_data=(y_data*1e5).reshape((10000,n))[:,2::m]####切片操作隔M列
scaler_q=StandardScaler()
normalized_q=scaler_q.fit_transform(y_data)

k_traintensor=torch.from_numpy(normalized_k[:9000,:])
q_traintensor=torch.from_numpy(normalized_q[:9000,:])
nreal=len(k_traintensor)
k_testtensor=torch.from_numpy((normalized_k[9000:10000,:]))
q_testtensor=torch.from_numpy((normalized_q[9000:10000, :]))
train_dataset=torch.utils.data.TensorDataset(q_traintensor,k_traintensor)
test_dataset=torch.utils.data.TensorDataset(q_testtensor, k_testtensor)
#
input_size=400
hidden_size=600
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
# 创建贝叶斯神经网络模型和优化器，超参数
# bayesian_net = BayesianNet(input_size,hidden_size,out_size,dropout_prob)
num_epochs =100
batch_size =10
num_qwt=400
learning_rate = 0.0001
optimizer =pyro.optim.DCTAdam({"lr":learning_rate})
criterion = nn.MSELoss()
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=1)


# 定义SVI对象
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
train_lossall=np.zeros(num_epochs)
# test_lossall=np.zeros(num_epochs)
# def predict(x, num_samples=200):
#     predictions = []
#     for _ in range(num_samples):
#         sampled_nn = guide(None, None, input_size, hidden_size, output_size)
#         pred = sampled_nn(x)
#         predictions.append(pred)
#     return torch.stack(predictions).mean(axis=0)

####训练模型
start_time=time.time()
for epoch in range(num_epochs):
    trainloss=0
    for i ,(inputs,labels) in enumerate(train_loader):
        inputs = inputs.float()
        # labels = torch.tensor(labels, dtype=torch.float32)
        labels=labels.float()
    # 计算损失并进行梯度优化
        loss = svi.step(inputs, labels,input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        trainloss+=loss
    train_lossall[epoch]=trainloss/len(train_loader)
    print("[iteration %04d] loss: %.4f" % (epoch + 1, trainloss/len(train_loader)))
    if epoch != 0 and train_lossall[epoch] < min(train_lossall[:epoch]) :
        pyro.get_param_store().save('bnn_model_params2.pt')
end_time=time.time()
execution_time = end_time - start_time
print("代码块执行时间：", execution_time/60, "min")

print('train_lossall',train_lossall)
# print('test_lossall',test_lossall)
plt.plot(range(num_epochs),train_lossall,label='train_loss')
# plt.plot(range(num_epochs),test_lossall,label='test_loss')
plt.title('normalized_k BNN_training')
plt.legend()
plt.show()

###预测函数

#
# q_test = torch.tensor(q_testtensor[0], dtype=torch.float32)
# k_pred= predict(q_test)
# k_pred= k_pred.detach().numpy()
# upperbound=np.percentile(k_pred, 97.5, axis=0)
# lowerbound=np.percentile(k_pred, 2.5, axis=0)
# x=range(400)
# plt.plot(x,upperbound,label="upperbound")
# plt.plot(x,lowerbound,label="lowerbound")
# plt.plot(x,np.array(k_testtensor[0]),label="ture_k")
# plt.plot(x,k_pred.mean(axis=0),label="pred_mean")
# plt.show()


