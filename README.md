1. Create Conda
  - `conda create -n g1test python=3.8`
  - `conda activate g1test`

2. Install Isaac Gym
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`

  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`


3. Other Dependencies
pip install tensorboard wandb setuptools==59.5  pydelatin pyfqmr tqdm opencv-python
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask onnx

cd legged_gym/legged_gym/scripts/
python train.py --exptid=xxx-xx-name
