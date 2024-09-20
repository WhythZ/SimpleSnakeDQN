# SimpleSnakeDQN

## License

This repo adopts [MIT License](https://spdx.org/licenses/MIT)

## About

A simple DQN snake game agent training programme

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

- After installing all the dependencies, run the `agent.py` to start training, the model will be saved in the `Model` folder
```bash
cd your_project_dir\Agent
python agent.py
```

- If you encounter the following error, install `nomkl` package in your conda environment
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
```
```bash
conda install nomkl
```