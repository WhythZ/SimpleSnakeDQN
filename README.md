# SimpleSnakeDQN

## License

This repo adopts [MIT License](https://spdx.org/licenses/MIT)

## About

A simple DQN snake game agent that you can train by yourself

## Environment

- Install pygame for game environment simulation
```
pip3/conda/pip install pygame
```

- Install pytorch for DQN model training, where CPU training is enough for this agent
```
pip3 install torch torchvision
```

- Install matplotlib and ipython to help display the training results
```
pip3/conda/pip install matplotlib ipython
```

- After installing all the dependencies, just run the `Agent.py` to start training, the model will be saved in the `Model` folder