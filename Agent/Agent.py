# 用于进行训练的智能体类

import torch
import random
import numpy as np
# 导入双端队列数据结构
from collections import deque
# 导入我们实现的游戏环境
from game import SnakeGameAI, Direction, Point
# 导入训练相关
from model import LinearQNet, QTrainer
# 导入可视化
from displayer import Plot
# 用于类型注解（标注指定长度的np.ndarray数组）
from typing import Literal

# 双端队列的最大容量
MAX_MEMORY = 100_000
# 批次大小
BATCH_SIZE = 1000
# Learning Rate
LR = 0.001
# 方块大小，与Game.py中的保持一致
BLOCK_SIZE = 20

class Agent:
    def __init__(self) -> None:
        # 用于记录游戏进行了几轮
        self.gameNumber = 0
        # 衡量随机度
        self.epsilon = 0
        # Discount Rate（小于1）
        self.gamma = 0.9
        # 使用双端队列进行记忆，使用popleft()进行出队
        self.memory = deque(maxlen=MAX_MEMORY)
        # 初始化训练模型的输入、隐藏、输出层
        self.model = LinearQNet(11, 256, 3)
        # 初始化训练器
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def GetState(self, game:SnakeGameAI) -> np.ndarray:
        # 获取蛇头坐标
        head = game.snake[0]
        # 获取蛇头的四周坐标，注意y轴正方向是朝向屏幕下方的
        headU = Point(head.x, head.y - BLOCK_SIZE)
        headD = Point(head.x, head.y + BLOCK_SIZE)
        headL = Point(head.x - BLOCK_SIZE, head.y)
        headR = Point(head.x + BLOCK_SIZE, head.y)
        # 以四个布尔值记录蛇的当前行进朝向
        dirU = (game.direction == Direction.UP)
        dirD = (game.direction == Direction.DOWN)
        dirL = (game.direction == Direction.LEFT)
        dirR = (game.direction == Direction.RIGHT)
        # 记录当前的状态并将其整合到如下列表内
        state = [
            # Danger Straight（处于如果直走就会导致游戏失败的状态）
            (dirU and game.IsCollision(headU)) or
            (dirD and game.IsCollision(headD)) or
            (dirL and game.IsCollision(headL)) or
            (dirR and game.IsCollision(headR)),

            # Danger Right（处于如果右转就会导致游戏失败的状态）
            (dirU and game.IsCollision(headR)) or
            (dirD and game.IsCollision(headL)) or
            (dirL and game.IsCollision(headU)) or
            (dirR and game.IsCollision(headD)),
            
            # Danger Left（处于如果左转就会导致游戏失败的状态）
            (dirU and game.IsCollision(headL)) or
            (dirD and game.IsCollision(headR)) or
            (dirL and game.IsCollision(headD)) or
            (dirR and game.IsCollision(headU)),
        
            # 四个布尔值记录的当前朝向的状态
            dirU,
            dirD,
            dirL,
            dirR,
        
            # 记录蛇头与食物的相对位置的状态
            game.food.y < game.head.y, #食物在头的上方
            game.food.y > game.head.y, #食物在头的下方
            game.food.x < game.head.x, #食物在头的左侧
            game.food.x > game.head.x  #食物在头的右侧
        ]
        # 将上述状态以numpy数组形式（元素类型为从bool转化而得的int类型）返回
        return np.array(state, dtype=int)
    
    def Remember(self, oldState:np.ndarray, lastAction:np.ndarray[Literal[3], int], reward:int, newState:np.ndarray, gameOver:bool) -> None:
        # 将传入的这一组信息以单个tuple的形式（注意此处有内层括号）写入智能体的memory队列内
        self.memory.append((oldState, lastAction, reward, newState, gameOver))
        # 如果记忆满了就popleft()删除队尾元素

    def TrainShortMemory(self, oldState:np.ndarray, lastAction:np.ndarray[Literal[3], int], reward:int, newState:np.ndarray, gameOver:bool) -> None:
        # 进行单组数据的训练
        self.trainer.StepTrain(oldState, lastAction, reward, newState, gameOver)

    def TrainLongMemory(self) -> None:
        # 对memory中的数组组数进行评估
        if len(self.memory) > BATCH_SIZE:
            # 从memory中随机抓取BATCH_SIZE组数据tuple进行训练（miniSample和memory一样是以tuple为元素的数组）
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            # 若是memory中的数据不足BATCH_SIZE组，则使用全部数据进行训练
            miniSample = self.memory
        
        # 进行多组数据的训练，需要将miniSample中的各组数据tuple的相同位置的项（借助zip()函数）打包传入
        oldStates, lastActions, rewards, newStates, gameOvers = zip(*miniSample)
        self.trainer.StepTrain(oldStates, lastActions, rewards, newStates, gameOvers)
        # for oldState, lastAction, reward, newState, gameOver in miniSample:
        #     self.trainer.StepTrain(oldState, lastAction, reward, newState, gameOver)

    def GetAction(self, state:np.ndarray) -> np.ndarray[Literal[3], int]:
        # 作为结果输出的动作，只需将三个元素位置之一换为1即可
        resultMoveAction = [0,0,0]
        
        # 当局数小于等于80（此处为了方便使用硬编码）局时，才有概率随机选取动作（局数越少，随机选取的概率越大）
        self.epsilon = 80 - self.gameNumber
        if random.randint(0,200) < self.epsilon:
            # 随机选取三个动作中的一个作为结果
            idx = random.randint(0,2)
            resultMoveAction[idx] = 1
        # 否则就是通过模型来预测下一步的动作
        else:
            # 使用torch
            state0 = torch.tensor(state, dtype=torch.float)
            # 这里得出的结果是浮点数组[float,float,float]，不能直接当作动作[1或0,1或0,1或0]使用
            prediction = self.model(state0)
            # 选取[float,float,float]三个元素中的最大的一个的位置索引（0、1、2）
            idx = torch.argmax(prediction).item()
            # 以最大的那个位置化为1作为输出动作
            resultMoveAction[idx] = 1
        
        # 输出结果动作
        return resultMoveAction

# 游戏主循环，并对智能体进行训练
def Train() -> None:
    # 存储得分相关信息
    totalScore = 0
    bestScore = 0
    # 用于可视化的数据
    plotScores = []
    plotMeanScores = []
    # 初始化智能体与游戏环境
    agent = Agent()
    game = SnakeGameAI()

    # 利用游戏环境进行训练的主循环
    while True:
        # 获取上一帧的状态
        oldState = agent.GetState(game)
        # 生成基于上一帧状态下，智能体应当采取的行动
        lastMoveAction = agent.GetAction(oldState)
        # 实际使用这个行动更新游戏，并存储相关信息（获取的奖励、游戏是否结束、以及游戏得分）
        reward, gameOver, currentScore = game.UpdateGame(lastMoveAction)

        # 获取当前状态（即执行了上一步行动后更新的游戏状态）用于分析而不是继续行动，每轮循环只会行动一次
        currentState = agent.GetState(game)
        # 使用目前为止获取到的信息进行短时记忆训练
        agent.TrainShortMemory(oldState, lastMoveAction, reward, currentState, gameOver)
        # 并存储到记忆（双端队列）中
        agent.Remember(oldState, lastMoveAction, reward, currentState, gameOver)

        # 游戏结束与否的判断
        if gameOver:
            # 重置游戏
            game.ResetGame()
            # 记录游戏轮次
            agent.gameNumber += 1
            # 进行长时记忆训练
            agent.TrainLongMemory()

            # 进行最高分数记录的更新
            if currentScore > bestScore:
                bestScore = currentScore
                # 当模型取得了更高的分数后，存储这个更优的模型
                agent.model.Save()

            # 进行信息的打印
            print('Game: ', agent.gameNumber, ' Score: ', currentScore, ' Record: ', bestScore)
            # 进行分数的计算
            plotScores.append(currentScore)
            totalScore += currentScore
            plotMeanScores.append(totalScore / agent.gameNumber)
            # 进行分数的可视化
            Plot(plotScores, plotMeanScores)

# 主函数开启训练
if __name__ == '__main__':
    Train()