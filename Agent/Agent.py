# 导入相关外部库
import torch
import random
import numpy as np
# 导入双端队列数据结构
from collections import deque
# 导入我们实现的游戏环境
from ..Game.Game import SnakeGameAI, Direction, Point

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
        # Discount Rate
        self.gamma = 0
        # 使用双端队列进行记忆，使用popleft()进行出队
        self.memory = deque(maxlen=MAX_MEMORY)

    def GetState(self, game) -> None:
        # 获取蛇头坐标
        head = game.snake[0]
        # 获取蛇头的四周坐标，注意y轴正方向是朝向屏幕下方的
        headU = Point(head.x, head.y - BLOCK_SIZE)
        headD = Point(head.x, head.y + BLOCK_SIZE)
        headL = Point(head.x - BLOCK_SIZE, head.y)
        headR = Point(head.x + BLOCK_SIZE, head.y)
        # 记录蛇的行进方向枚举，以简化代码
        dirU = Direction.UP
        dirD = Direction.DOWN
        dirL = Direction.LEFT
        dirR = Direction.RIGHT
    
    def Remember(self, oldState, lastAction, reward, newState, gameOver) -> None:
        pass

    def TrainLongMemory(self) -> None:
        pass

    def TrainShortMemory(self,  oldState, lastAction, reward, newState, gameOver) -> None:
        pass

    def GetAction(self, state) -> None:
        pass

# 游戏主循环，并对智能体进行训练
def Train() -> None:
    # 存储得分相关信息
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    bestScore = 0
    # 初始化智能体与游戏环境
    agent = Agent()
    game = SnakeGameAI()

    # 训练主循环，内嵌游戏主循环
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

            # 进行信息的打印
            print('Game: ', agent.gameNumber, ' Score: ', currentScore, ' Record: ', bestScore)

            

# 主函数开启训练
if __name__ == '__main__':
    Train()