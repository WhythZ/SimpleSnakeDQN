# 用于以图表形式可视化显示训练成果

import matplotlib.pyplot as plt
from IPython import display

def Plot(scores, meanScores):
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Total Game Rounds')
    plt.ylabel('Scores')
    plt.plot(scores)
    plt.plot(meanScores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1]))
    plt.show()
