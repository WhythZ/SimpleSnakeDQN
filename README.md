# SimpleSnakeDQN

## License
This repo adopts [MIT License](https://spdx.org/licenses/MIT)

## About
This repo stores a simple DQN snake game agent training program

![demo.gif](/Resource/Demo.gif)

## Environment
- Install pygame for game environment simulation

```bash
pip3/conda/pip install pygame
```

- Install pytorch for DQN model training, where CPU training is enough for this agent

```bash
pip3 install torch torchvision
```

- Install matplotlib and ipython to help display the training results

```bash
pip3/conda/pip install matplotlib ipython
```

- After installing all the dependencies, run the `agent.py` to start new training task

```bash
cd your_project_dir\Agent
python Agent.py
```

- If you encounter the following error, install `nomkl` package in your conda environment

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
```

```bash
conda install nomkl
```

## Improvement
- The snake agent will frequently crash into its own body then end the game, the reason is that the agent is unaware of the position info about the snake body (because of the state design), thus unconscious to reduce the risk of crashing into itself
