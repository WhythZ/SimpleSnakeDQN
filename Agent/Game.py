# 贪吃蛇游戏环境模拟程序，用于智能体训练
# 可用动作：[1,0,0]直走，[0,1,0]右转，[0,0,1]左转
# 状态设置：以np.ndarray数组形式存在，参考智能体程序内的的相关代码
# 奖励设置：吃到食物则奖励reward+10，游戏结束（发生碰撞或者持续长时间不得分）则惩罚reward-10，其余行为不得分不扣分

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
# 用于类型注解（标注指定长度的np.ndarray数组）
from typing import Literal

# 记录颜色（RGB）以便使用
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
# 方块大小与游戏速度
BLOCK_SIZE = 20
SPEED = 20

# 初始化pygame
pygame.init()
# 载入字体
font = pygame.font.Font("Resource/Arial.ttf", 25)

# 蛇的四个前进方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 自定义Point的元组数据类型，每个该类型对象拥有一对(x,y)坐标
Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    # 初始化贪吃蛇游戏
    def __init__(self, w:int=640, h:int=480) -> None:
        # 游戏窗口的宽高
        self.w = w
        self.h = h
        # 初始化图像显示
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        # 全局计时器，即时游戏重新开始也不会停止计时
        self.clock = pygame.time.Clock()

        # 重置用于训练智能体的游戏环境，在执行此函数前进行的其它初始化都是静态性质的，无需每次都重置
        self.ResetGame()
        
    def ResetGame(self) -> None:
        # 初始化蛇的初始前进方向
        self.direction = Direction.RIGHT
        
        # 初始化分别代表蛇的头以及整个身体的两个成员变量
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)] 
        
        # 初始化用于记录本局游戏得分数的成员变量
        self.score = 0
        # 初始化食物位置
        self.food = None
        self.SummonFood()

        # 初始化一个计数器，用于记录游戏运行了多少帧，本质是当作计时器使用
        self.frameIteration = 0

    def UpdateGame(self, action:np.ndarray[Literal[3], int]) -> tuple[int, bool, int]:
        # 递增游戏运行的总帧数，只有当调用ResetGame后才会清零
        self.frameIteration += 1
        
        # 检测事件输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # # 玩家输入事件的接收
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN
        
        # 在蛇头位置新插入一个放块，以模拟蛇每帧的前进
        self.Move(action)
        self.snake.insert(0, self.head)
        
        # 检测游戏是否结束，将需要返回的值先声明关键字
        reward = 0
        gameOver = False
        
        # 如果发生了碰撞，或者长时间（与身体长度成正比）不得分，则进行惩罚
        if self.IsCollision() or self.frameIteration >= 100*len(self.snake):
            # 游戏结束进行惩罚
            reward = -10
            gameOver = True
            # 直接返回惩罚、游戏结束的信息、以及得分
            return reward, gameOver, self.score
        # 如果吃到了食物则加分、进行奖励、并生成新的食物，否则一直移动
        elif self.head == self.food:
            self.score += 1
            self.SummonFood()
        # 否则删除尾部元素，然后在下一个循环开头会增长头部，以达成移动
        else:
            self.snake.pop()
        
        # 更新分数与计时器
        self.UpdateUI()
        self.clock.tick(SPEED)

        # 返回奖励、游戏结束与否（吃到了食物的话返回的是初始值False）、阶段性得分数
        return reward, gameOver, self.score

    def SummonFood(self) -> None:
        # 在地图上随机位置生成食物
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # 
        if self.food in self.snake:
            self.SummonFood()

    def IsCollision(self, pt:Point=None) -> bool:
        # pt代表一个点，初始化为蛇头
        if pt is None:
            pt = self.head
        # 蛇头碰触到边界
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 蛇头碰触到自己的蛇身
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def UpdateUI(self) -> None:
        self.display.fill(BLACK)
        
        # 蛇移动的图像渲染
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 分数文字的渲染
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def Move(self, action:np.ndarray[Literal[3], int]) -> None:
        # 记录顺时针方向，用于得到[当前方向]叠加[action]操作后得到的具体方向
        clockWise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # 保存蛇原本的方向在clockWise中的索引数，类似STL的vector的find函数
        idx = clockWise.index(self.direction)
        # 用于存储新方向
        new_dir = None

        # 对比传入的action动作矩阵，[1,0,0]表示直走，[0,1,0]表示右转，[0,0,1]表示左转
        if np.array_equal(action, [1,0,0]):
            # 若是直走，则蛇保持原有的方向不变
            new_dir = clockWise[idx]
        if np.array_equal(action, [0,1,0]):
            # 若是向右转，则在原方向的基础上顺时针变换一个方向，取模应对的是idx从最大索引3递增到4的情况
            new_dir = clockWise[(idx + 1) % 4]
        if np.array_equal(action, [0,0,1]):
            # 若是向左转，则在原方向的基础上逆时针变换一个方向，(-1%4)的结果是3
            new_dir = clockWise[(idx - 1) % 4]

        # 赋予当前蛇以新的方向
        self.direction = new_dir

        # 移动；注意屏幕的下方是y轴的正方向（SDL、EasyX等图形API也是这个毛病），所以向下是对y坐标递增
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

# # 使得游戏响应玩家输入
# if __name__ == '__main__':
#     # 初始化贪吃蛇游戏环境
#     game = SnakeGameAI()
#     # 游戏主循环
#     while True:
#         # 每循环一次调用一次更新，并获取返回值信息
#         gameOver, score = game.UpdateGame()
#         # 游戏结束即跳出游戏主循环
#         if gameOver == True:
#             break
#     # 在游戏结束后打印总得分数
#     print('Final Score', score)
#     # 销毁游戏环境
#     pygame.quit()