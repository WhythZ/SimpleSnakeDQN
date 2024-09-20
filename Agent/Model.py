# 模型用于存储智能体训练结果的相关信息，并依据这些数据提供一些预测的方法

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 用于以文件形式存储模型
import os
# 用于类型注解
import numpy as np
from typing import Literal

# 继承自nn
class LinearQNet(nn.Module):
    # 接收神经网络的输入层、中间隐藏层、输出层的尺寸
    def __init__(self, inputSize:int, hiddenSize:int, outputSize:int) -> None:
        # 调用基类的初始化函数
        super().__init__()
        # 从输入层到隐藏层
        self.linearLayer01 = nn.Linear(inputSize, hiddenSize)
        # 从隐藏层到输出层
        self.linearLayer02 = nn.Linear(hiddenSize, outputSize)

    # 重写基类的forward()函数
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linearLayer01(x))
        x = self.linearLayer02(x)
        return x
    
    # 用于保存模型
    def Save(self, fileName='model.pth'):
        # 存储模型的文件夹路径
        modelFolderPath = './Model'
        # 如果路径不存在则创建
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        # 以传入的文件名，将模型（只保存模型的参数）保存到目录下
        finalFileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), finalFileName)

    # # 用于加载已有模型
    # def Load(self) -> None:
    #     modelPath = './Model/model.pth'

class QTrainer:
    def __init__(self, model:LinearQNet, lr:float, gamma:float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 损失函数判据，使用均方误差（Mean Square Error）即loss = (Q_new - Q)^2进行赋值
        self.criterion = nn.MSELoss()

    # 当进行短时记忆训练时接收单组数据输入；当进行长时记忆训练时，接收多组相同长度的数据输入
    def StepTrain(self, oldState:np.ndarray, lastAction:np.ndarray[Literal[3], int], reward:int, newState:np.ndarray, gameOver:bool) -> None:
        # 将传入数据进行类型转换
        oldState = torch.tensor(oldState, dtype=torch.float)
        newState = torch.tensor(newState, dtype=torch.float)
        lastAction = torch.tensor(lastAction, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n,x)

        if len(oldState.shape) == 1:
            # (1,x)
            oldState = torch.unsqueeze(oldState, 0)
            newState = torch.unsqueeze(newState, 0)
            lastAction = torch.unsqueeze(lastAction, 0)
            reward = torch.unsqueeze(reward, 0)
            # 只有一个元素的元组
            gameOver = (gameOver, )
        
        # 第一步：通过当前状态预测Q值
        Q = self.model(oldState)
        # https://youtu.be/L8ypSXwyBds?si=94JyvLTXb2n7HOhv&t=5338
        target = Q.clone()
        for idx in range(len(gameOver)):
            Q_new = reward[idx]
            # 第二步：如果游戏失败了，则进行Q_new的计算
            if not gameOver[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(newState[idx])) 
            target[idx][torch.argmax(lastAction).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, Q)
        loss.backward()

        self.optimizer.step()