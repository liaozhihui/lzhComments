import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,TensorDataset,DataLoader
import torch.nn.functional as F


torch.manual_seed(1)

#hyper parameter
Learning_rate = 0.01
Batch_size = 32
Epoch =16

# fake dataset
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y = x.pow(2)+0.1*torch.normal(torch.zeros(x.size()[0], 1), torch.ones(x.size()[0], 1))

# plot dataset
# plt.scatter(x.numpy(),y.numpy())
# plt.show()

torch_dataset = TensorDataset(x,y)
loader = DataLoader(dataset=torch_dataset,batch_size=Batch_size,shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
net_Adagrad = Net()

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam, net_Adagrad]

# 创建不同的优化器用来训练不同的网络
opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=Learning_rate)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=Learning_rate,momentum=0.8,nesterov=True)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=Learning_rate,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=Learning_rate,betas=(0.9,0.99))
opt_Adagrad = torch.optim.Adagrad(net_Adagrad.parameters(),lr=Learning_rate)

optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam,opt_Adagrad]

criterion = torch.nn.MSELoss()
losses_his = [[],[],[],[],[]]  # 记录 training 时不同神经网络的 loss

for epoch in range(Epoch):

    for step, (b_x, b_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = criterion(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())

        if step % 25 == 1 and epoch % 7 == 0:
            labels = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'Adagrad']
            for i, l_his in enumerate(losses_his):
                plt.plot(l_his, label=labels[i])
            plt.legend(loc='best')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.ylim((0, 0.2))
            plt.xlim((0, 200))
            print('epoch: {}/{},steps:{}/{}'.format(epoch + 1, Epoch, step * Batch_size, len(loader.dataset)))
            plt.show()

