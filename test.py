import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim  import DCTAdam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size =10
x_data = np.zeros((1, 400)).astype('float32')
for root, dirs, files in os.walk('sgsimdata'):
    for file in files:
        new = np.loadtxt(os.path.join(root, file)).T
        new = torch.from_numpy(new).to(device)

        # Convert tensor to NumPy
        numpy_array = new.cpu().numpy()

        # Concatenate NumPy arrays
        x_data = np.concatenate((x_data, numpy_array), axis=0)



y_data = np.loadtxt("qwt_nonoise_change_sgsim10000.txt").astype('float32')
m=10
n=4000

y_data = (y_data*1e5).reshape((10000,n))[:,2::m]
x_data = torch.from_numpy(x_data).to(device)
y_data = torch.from_numpy(y_data).to(device)

q_traintensor=y_data[:9000,:].to(device)
k_traintensor=x_data[:9000,:].to(device)
train_dataset=torch.utils.data.TensorDataset(q_traintensor,k_traintensor)

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size)

input_size=400
hidden_size=250
output_size=400
# 定义贝叶斯神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super( SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x =self.fc2(x)
        return x

nnmodel = SimpleNN(input_size, hidden_size, output_size).to(device)

def model(x, y, input_size, hidden_size, output_size):
    fc1_weight_prior = dist.Normal(
        torch.zeros(hidden_size, input_size).to(device), torch.ones(hidden_size, input_size).to(device)
    ).to_event(2)
    fc1_bias_prior = dist.Normal(torch.zeros(hidden_size).to(device), torch.ones(hidden_size).to(device)).to_event(1)
    fc2_weight_prior = dist.Normal(
        torch.zeros(output_size, hidden_size).to(device), torch.ones(output_size, hidden_size).to(device)
    ).to_event(2)
    fc2_bias_prior = dist.Normal(torch.zeros(output_size).to(device), torch.ones(output_size).to(device)).to_event(1)

    priors = {
        "fc1.weight": fc1_weight_prior,
        "fc1.bias": fc1_bias_prior,
        "fc2.weight": fc2_weight_prior,
        "fc2.bias": fc2_bias_prior,
    }

    lifted_module = pyro.random_module("module", nnmodel, priors)
    lifted_nn = lifted_module()

    with pyro.plate("data", x.shape[0]):
        prediction_mean = lifted_nn(x)
        pyro.sample("obs",
                    dist.MultivariateNormal(prediction_mean, torch.eye(output_size).to(device)).to_event(1),
                    obs=y.to(device))

def guide(x, y, input_size, hidden_size, output_size):
    fc1_weight_loc = pyro.param("fc1_weight_loc", torch.randn(hidden_size, input_size).to(device))
    fc1_weight_scale = pyro.param("fc1_weight_scale", torch.ones(hidden_size, input_size).to(device), constraint=dist.constraints.positive)
    fc1_bias_loc = pyro.param("fc1_bias_loc", torch.randn(hidden_size).to(device))
    fc1_bias_scale = pyro.param("fc1_bias_scale", torch.ones(hidden_size).to(device), constraint=dist.constraints.positive)

    fc2_weight_loc = pyro.param("fc2_weight_loc", torch.randn(output_size, hidden_size).to(device))
    fc2_weight_scale = pyro.param("fc2_weight_scale", torch.ones(output_size, hidden_size).to(device), constraint=dist.constraints.positive)
    fc2_bias_loc = pyro.param("fc2_bias_loc", torch.randn(output_size).to(device))
    fc2_bias_scale = pyro.param("fc2_bias_scale", torch.ones(output_size).to(device), constraint=dist.constraints.positive)

    variational_dist = {
        "fc1.weight": dist.Normal(fc1_weight_loc, fc1_weight_scale).to_event(2),
        "fc1.bias": dist.Normal(fc1_bias_loc, fc1_bias_scale).to_event(1),
        "fc2.weight": dist.Normal(fc2_weight_loc, fc2_weight_scale).to_event(2),
        "fc2.bias": dist.Normal(fc2_bias_loc, fc2_bias_scale).to_event(1),
    }
    lifted_module = pyro.random_module("module", nnmodel, variational_dist)
    lifted_nn = lifted_module()

num_epochs =150
batch_size =10
num_qwt=400
learning_rate = 0.001

import torch.optim as optim
optimizer = optim.Adam(nnmodel.parameters(), lr=learning_rate)
optimizer = pyro.optim.PyroOptim(optimizer, optim_args={"lr": learning_rate})
criterion = nn.MSELoss()

train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
train_lossall=np.zeros(num_epochs)


def predict(x, num_samples=200):
    x = x.to(device)

    predictions = []
    for _ in range(num_samples):
        sampled_nn = guide(None, None, input_size, hidden_size, output_size)
        pred = sampled_nn(x)
        predictions.append(pred)
    return torch.stack(predictions).mean(axis=0)


for epoch in range(num_epochs):
    trainloss=0
    for inputs, labels, *_ in train_loader:
        inputs, labels = inputs.float(), labels.float()
        inputs, labels = inputs.to(device), labels.to(device)
        loss = svi.step(inputs, labels, input_size, hidden_size, output_size)
        # loss = svi.step(inputs, labels, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        trainloss += loss
    train_lossall[epoch] = trainloss / len(train_loader)
    print("[iteration %04d] loss: %.4f" % (epoch + 1, trainloss / len(train_loader)))
    if epoch != 0 and train_lossall[epoch] < min(train_lossall[:epoch]):
        pyro.get_param_store().save('bnn_model_params_cuda.pt', device=device)


print('train_lossall',train_lossall)
plt.plot(range(num_epochs),train_lossall,label='train_loss')

plt.title('k BNN_training')
plt.legend()
plt.show()




