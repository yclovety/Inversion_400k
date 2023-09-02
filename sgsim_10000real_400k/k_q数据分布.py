import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# x_data=np.zeros((1,400))
# for root, dirs, files in os.walk('sgsimdata'):
#     for file in files:
#         # print(os.path.join(root, file))
#         new =np.loadtxt(os.path.join(root, file)).T
#         x_data=np.concatenate((x_data,new))
# x_data=np.delete(x_data,0,axis=0)
# scaler_k=MinMaxScaler()
# # normalized_k=scaler_k.fit_transform(x_data)
#
y_data=np.loadtxt("qwt_nonoise_change_sgsim10000.txt")
m=10
n=4000
y_data=(y_data*1e5).reshape((10000,n))####切片操作隔M列
# scaler_q=MinMaxScaler()
# normalized_q=scaler_q.fit_transform(y_data)
# y_inverse=scaler_q.inverse_transform(normalized_q)
# 1绘制密度图
sns.kdeplot(y_data[:,0], shade=True,label='q0')
sns.kdeplot(y_data[:,2],shade=True,label='q2')
# plt.title('q_standard')
plt.title('q')
plt.legend()
plt.show()

# 2.箱线图箱线图的箱体表示数据的四分位数范围，箱体中的横线表示数据的中位数， 箱体上下的线条表示数据的最大值和最小值，
# 箱体外的点表示数据中的异常值。
# fig, ax = plt.subplots()
# ax.boxplot(x_data[:,:3], vert=False)
# col_std=np.std(normalized_k,axis=0)
# plt.plot(range(400),col_std)
# plt.show()


