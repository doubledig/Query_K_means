import random
import time

import torch
import torchvision.datasets

if __name__ == '__main__':
    # 前置设置
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    kc = 10  # 聚类中心数
    delta = 0.2
    epsilon = 0.2  # 超参
    # 使用torchvision提供的包读取MNIST数据集并进行处理
    # 训练集
    s_time = time.time()
    mnist = torchvision.datasets.MNIST('data')
    train_data_x = mnist.data.to(device=device, dtype=torch.float).reshape(60000, -1)
    train_data_y = mnist.targets.to(device=device, dtype=torch.float)
    # 测试集
    mnist = torchvision.datasets.MNIST('data', train=False)
    test_data_x = mnist.data.to(device=device, dtype=torch.float).reshape(10000, -1)
    test_data_y = mnist.targets.to(device=device, dtype=torch.float)
    del mnist
    print('- load dataset in {:.3f} -'.format(time.time()-s_time))
    # 开始训练
    print('- start training -')
    '''
        原始K-means，mnist数据集使用原始K-means效果并不好，多训练几次取标签没有重复的结果
    '''
    print('- formal k-means train-')
    s_time = time.time()
    # 随机确定聚类中心
    hp = set()
    while len(hp) < 10:
        c_h = torch.randint(train_data_x.size(0), size=(kc, 1), dtype=torch.long, device=device)
        c_h = train_data_x[c_h[:, 0], :]
        c = torch.zeros(kc, (train_data_x.size(1)), device=device)
        while not torch.equal(c, c_h):
            c = c_h.clone()
            # 进行分类，得到cl = 60000
            cl = torch.argmin(torch.cdist(train_data_x, c_h), dim=1)
            # 重新计算中心
            for i in range(kc):
                index = torch.eq(cl, i)
                c_h[i, :] = train_data_x[index, :].sum(dim=0) / index.sum()
        # 聚类中心打标签，
        cl = torch.argmin(torch.cdist(train_data_x, c_h), dim=1)
        hp = set()
        for i in range(kc):
            index = torch.eq(cl, i)
            h = 0
            a = 0
            for j in range(kc):
                b = torch.eq(train_data_y[index], j).sum()
                if b > a:
                    a = int(b)
                    h = j
            hp.add(h)
            c[h, :] = c_h[i, :]
    print(hp)  # 输出判断的标签
    print('- k-means cluster centers processed in {:.3f} -'.format(time.time() - s_time))
    '''
        query K-means
    '''
    print('- query k-means train-')
    s_time = time.time()
    num_list = [i for i in range(train_data_x.size(0))]
    random.shuffle(num_list)
    x = []
    x_num = []
    for i in range(kc):
        x.append([])
        x_num.append(0)
    min_c = kc/(epsilon*delta)
    i = 0
    while min(x_num) < min_c:
        h = num_list[i]
        x[int(train_data_y[h])].append(train_data_x[h, :])
        x_num[int(train_data_y[h])] += 1
        i += 1
    c_q = torch.zeros(kc, (train_data_x.size(1)), device=device)
    for i in range(kc):
        for j in x[i]:
            c_q[i, :] += j
        c_q[i, :] /= x_num[i]
    print('- query k-means cluster centers processed in {:.3f} -'.format(time.time() - s_time))
    # 性能比较
    print('- performance -')
    print('- formal k-means test -')
    cl = torch.argmin(torch.cdist(test_data_x, c), dim=1)
    acc = torch.eq(test_data_y, cl).sum() / test_data_y.size(0) * 100
    print('- formal k-means accuracy is {:.2f}% -'.format(acc))
    print('- query k-means test -')
    cl = torch.argmin(torch.cdist(test_data_x, c_q), dim=1)
    acc = torch.eq(test_data_y, cl).sum() / test_data_y.size(0) * 100
    print('- query k-means accuracy is {:.2f}% -'.format(acc))
    pass
