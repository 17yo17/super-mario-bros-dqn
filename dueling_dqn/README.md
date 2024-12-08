# super-mario-rl-agent
A reinforcement learning implementation for super mario bros. using gym-super-mario-bros

## Prerequisites
 - This project was developed with Python 3.11.9, but any version between 3.9.x and 3.11.x should work
 - Follow [PyTorch's](https://pytorch.org/get-started/locally/) instructions to set up the PyTorch library on your machine, as the steps are different according to what CUDA version you are using, if any
 - Install the exact requirements using `pip install -r requirements.txt`

## Running the program
You can pass various arguments to `main.py` in order to train or run a model.

### Training a new model
Use `main.py --train` will train a model and save to the `./models/` directory using a timestamp as a name.
If you want to change the number of iterations, modify the MAX_ITER global value in `main.py`.

### Testing an existing model
Use `main.py --test --model model_name.pth` to test a model. For PPO models use `main.py --test --model model_name` with no extension
If you want to change the number of episodes to test with, modify the TEST_EPISODES global value in `main.py`

### Options
When running main, you can pass the following options to activate certain features:
- Pass `--display` to visualize the gameplay on screen as the model is trained/tested
- Pass `--record_video` to record the gameplay to an mp4 file
- Note that Display and Video shouldn't be used together