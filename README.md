# From Mystery to Mastery: Failure Diagnosis for Improving Manipulation Policies

We introduce RoboMD a deep reinforcement learning-based framework designed to identify failure modes in robotic manipulation policies. By simulating diverse conditions and quantifying failure probabilities, RoboFail provides insights into model robustness and adaptability.

## Installation

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- CUDA (if using GPU)
- Conda (recommended for managing environments)


### Setting Up the Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/Robo-MD/Robo-MD-RSS.github.io.git
   cd Robo-MD-RSS.github.io
   ```

2. Create a Conda environment:
   ```bash
   conda create --name robomd python=3.8 -y
   conda activate robomd
   ```

### Installing Dependencies
#### 1. **Install robosuite**
   ```bash
   pip install robosuite
   ```


#### 2. **Install robomimic**
   ```bash
   git clone https://github.com/ARISE-Initiative/robomimic.git
   cd robomimic
   pip install -e .
   ```

#### 3. **Additional Dependencies**
   Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── configs/               # Configuration files for actions and training
├── env/                   # Environment implementations
├── scripts/               
├── utils/                 # Utility functions (e.g., loss computations)
├── train_continuous.py     # Training script for continuous latent actions
├── train_discrete.py       # Training script for discrete latent actions
├── train_embedding.py      # Training script for embedding learning
├── README.md               # Project documentation
├── requirements.txt        # Required dependencies
```

---

## Usage

### Training with Continuous Actions

To train an RL policy using a latent action space, run:
```bash
python train_continuous.py --name <run_name> --task <task_name> --agent <path_to_agent> --rl_timesteps 3000
```
Example:
```bash
python train_continuous.py --name latent_rl --task lift --agent models/bc_agent.pth --rl_timesteps 50000
```

### Training Discrete Action Policies

For training RL with a discrete action space:
```bash
python train_discrete.py --name <run_name> --task <task_name> --agent <path_to_agent> --rl_timesteps 3000
```

### Training Embeddings

To train and store known embeddings:
```bash
python train_embedding.py --path <dataset_path>
```
This script extracts embeddings from a dataset and stores them in an HDF5 file.







## License
MIT License © 2024 RoboMD Team
