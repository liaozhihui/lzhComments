import torch
import time
from torch import nn
# from models.LSTMAttention import BiLSTM_Attention,vocab_size
from models.TEXTCNN import TextCNN,vocab_size
from prepo import train_iter,val_iter,text
import random
import numpy as np
import os
random.seed(15)
np.random.seed(15)
torch.manual_seed(15)
vocab = text.vocab
def evaluate_accuracy(data_iter,net,save=False):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label

            X = X.permute(1, 0)
            y.data.sub_(1)
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    global_step = 0
    batch_count = 0
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, batch in enumerate(train_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  #因为label中多了unk所以要减一
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
            global_step+=1
        test_acc = evaluate_accuracy(test_iter, net)
        if test_acc>best_test_acc:
            best_test_acc = test_acc
            saveDir = net._get_name()
            output_dir = os.path.join("result", saveDir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(net.state_dict(), os.path.join(output_dir,"model.bin"))
            with open(os.path.join(output_dir,"acc_result.txt"),"w") as f:
                f.write(f"acc:{best_test_acc}")

        print(
            'epoch %d, loss %.4f, train acc %.5f, test acc %.5f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))

def main():
    lr, num_epochs = 0.001, 15
    embedding_dim, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(vocab_size, embedding_dim, kernel_sizes, num_channels)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([3,1,6])).float()) #0,2,1
    train(train_iter, val_iter, net, loss, optimizer, num_epochs)


if __name__ == '__main__':
    main()