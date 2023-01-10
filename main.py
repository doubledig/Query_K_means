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
    # 使用torchvision提供的包读取MNIST数据集并进行处理
    # 训练集
    s_time = time.time()
    mnist = torchvision.datasets.MNIST('data')
    train_data_x = mnist.data.to(device=device, dtype=torch.int16).reshape(60000, -1)
    train_data_y = mnist.targets.to(device=device, dtype=torch.int8)
    # 测试集
    mnist = torchvision.datasets.MNIST('data', train=False)
    test_data_x = mnist.data.to(device=device, dtype=torch.int16).reshape(10000, -1)
    test_data_y = mnist.targets.to(device=device, dtype=torch.int8)
    del mnist
    print('- load dataset in {:.3f} -'.format(time.time()-s_time))
    # 开始训练
    print('- start training -')
    # 原始K-means
    print('- formal k-means -')
    s_time = time.time()
    # 随机确定聚类中心
    c_h = torch.rand((train_data_x.size(1), kc), device=device) * 255
    c = torch.zeros((train_data_x.size(1), kc), device=device)
    # Query K-means
    # 性能比较
    pass
