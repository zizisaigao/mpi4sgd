# 🚕 DSA5208 Project 1: MPI Parallel Neural Network Training (New York Taxi Data)

> Use MPI to parallelize model training on the large-scale New York taxi dataset `nytaxi2022.csv`, supporting batch experiments with multiple parameter combinations. A version of the dataset can be found on kaggle (https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers), and the real dataset used in the project similar to that but much larger.

---

## 🧰 Project Dependencies

- **Operating System**: Windows Subsystem for Linux (WSL)
- **Python 3.x**
- **MPI Environment** (accessed via `mpi4py`)
- **Python Libraries**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `mpi4py`

---

## 🚀 Quick Start Guide

### Launch WSL and Navigate to the Project Directory
```bash
wsl
cd /path/to/your/project-directory
```

### Create and Activate the Environment
```bash
python3 -m venv ~/dsa5208
source ~/dsa5208/bin/activate
```

### Install Required Python Packages
```bash
pip install mpi4py matplotlib numpy scikit-learn pandas tqdm
```

### Data Preprocessing
First, place `nytaxi2022.csv` in the current directory.
```bash
python preprocess.py --input_file nytaxi2022.csv
```
Afterwards, the required three `.npz` files will appear in the `data` folder.

### Single Training Run
```bash
mpirun -n 4 python nn_mpi.py --hidden 64 --batch 512 --activ sigmoid
```
Use the `-n` option to specify the number of processes to launch.

### Batch Experiments: Running All Parameter Combinations
All logs and results will be printed to `mpi_train_log.txt`. Meanwhile, the results will be saved to the file `results.csv`. The visualization files will be saved in the `figures/` directory.
```bash
./run_all_mpi.sh
```

### 📁 File Structure
```bash
data/
├── scaler_all.npz
├── test_all.npz
└── train_all.npz
figures/                    
├── mpi_loss_relu_h64_b32_p4.png
├── mpi_loss_tanh_h128_b64_p4.png
└── ...
mpi_train_log.txt
nn_mpi.py
nytaxi2022.csv
preprocess.py
hostfile
README.md
results.csv
run_all_mpi.sh

```

