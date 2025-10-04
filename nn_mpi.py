import os, sys, time, argparse, numpy as np
from mpi4py import MPI
from typing import Tuple
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- activation function --------------------
def sigma(x, name: str):
    if name == 'relu':   
        return np.maximum(0, x)
    if name == 'tanh':   
        return np.tanh(x)
    if name == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    raise ValueError(name)

def dsigma(x, name: str):
    if name == 'relu':   
        return (x > 0).astype(float)
    if name == 'tanh':   
        return 1 - np.tanh(x) ** 2
    if name == 'sigmoid':
        s = sigma(x, 'sigmoid')
        return s * (1 - s)
    raise ValueError(name)

# -------------------- network initialization --------------------
def init_params(m: int, n: int) -> np.ndarray:
    theta = np.empty(n * (m + 2) + 1, dtype=np.float32)
    # He: std = sqrt(2 / fan_in)
    std1 = np.sqrt(2.0 / (m + 1)) 
    std2 = np.sqrt(2.0 / (n + 1))
    theta[:n*(m+1)] = np.random.normal(0, std1, n*(m+1)).astype(np.float32)
    theta[n*(m+1):] = np.random.normal(0, std2, n+1).astype(np.float32)
    return theta

def unpack_theta(theta: np.ndarray, m: int, n: int):
    W1 = theta[:n * (m + 1)].reshape(m + 1, n)
    w2 = theta[n * (m + 1):]
    return W1[:-1, :], W1[-1, :], w2[:-1], w2[-1]

# -------------------- forward + gradient --------------------
def forward(X: np.ndarray, theta: np.ndarray, m: int, n: int, activ: str):
    W1, b1, w2, b2 = unpack_theta(theta, m, n)
    z1 = X @ W1 + b1
    a1 = sigma(z1, activ)
    a1_1 = np.column_stack([a1, np.ones(a1.shape[0], dtype=np.float32)])
    y_hat = a1_1 @ np.append(w2, b2)
    return y_hat, z1, a1, w2

def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
             m: int, n: int, activ: str) -> Tuple[np.ndarray, float]:
    B = X.shape[0]
    y_hat, z1, a1, w2 = forward(X, theta, m, n, activ)
    delta2 = (y_hat - y.ravel())[:, None]
    a1_1 = np.column_stack([a1, np.ones(a1.shape[0], dtype=np.float32)])
    grad_w2 = (delta2 * a1_1).mean(axis=0)          # (n+1,)
    ds = dsigma(z1, activ)
    delta1 = delta2 * w2 * ds
    grad_W = (X.T @ delta1) / B                     # (m, n)
    grad_b = delta1.mean(axis=0)                    # (n,)
    grad = np.empty_like(theta)
    grad[:m*n] = grad_W.reshape(-1)
    grad[m*n: (m+1)*n] = grad_b
    grad[(m+1)*n:] = grad_w2
    loss = 0.5 * ((y_hat - y.ravel()) ** 2).mean()
    # grad norm
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1.0:
        grad *= 1.0 / grad_norm
    return grad, loss

# -------------------- parallel RMSE --------------------
def parallel_rmse(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                  m: int, n: int, activ: str, y_scale: float, y_min: float):
    """Perform local prediction chunk by chunk, reduce the sum of squares, and avoid allocating a huge matrix all at once."""
    local_N = X.shape[0]
    # Avoid performing a forward pass on a huge matrix all at once.
    batch = 10000
    local_sq = 0.0
    for start in trange(0, local_N, batch):
        end = min(start + batch, local_N)
        Xb, yb = X[start:end], y[start:end]
        y_hat, _, _, _ = forward(Xb, theta, m, n, activ)
        # print(f"[DEBUG] Rank {comm.Get_rank()}: Xb={Xb.shape}, yb={yb.shape}, y_hat={y_hat.shape}")
        y_hat = y_hat * y_scale + y_min
        y_true = yb * y_scale + y_min
        local_sq += ((y_hat - y_true) ** 2).sum()
    # allreduce
    global_sq = comm.allreduce(local_sq, op=MPI.SUM)
    global_cnt = comm.allreduce(np.array([y.size], dtype=np.float64), op=MPI.SUM)
    return np.sqrt(global_sq / global_cnt).item()

# -------------------- data split --------------------
def scatter_data(X: np.ndarray, y: np.ndarray):
    """
    Process 0 evenly splits X and y by the number of samples, and other processes receive their local chunks.  
    Returns (local_X, local_y, counts, displs).
    """
    if rank == 0:
        N = X.shape[0]
        chunk = (N + size - 1) // size
        counts = [chunk] * size
        counts[-1] = N - chunk * (size - 1)
        displs = [i * chunk for i in range(size)]

        X_send = X.astype(np.float32, copy=False)
        y_send = y.astype(np.float32, copy=False)
    else:
        counts = displs = None
        X_send = None
        y_send = None

    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)
    local_N = counts[rank]
    feat_dim = X.shape[1] if rank == 0 else None
    feat_dim = comm.bcast(feat_dim, root=0)

    local_X = np.zeros((local_N, feat_dim), dtype=np.float32)
    local_y = np.zeros((local_N, 1), dtype=np.float32)
    comm.Scatterv([X_send, counts, displs, MPI.FLOAT], local_X, root=0)
    comm.Scatterv([y_send, counts, displs, MPI.FLOAT], local_y, root=0)

    # check again
    if not np.all(np.isfinite(local_X)) or not np.all(np.isfinite(local_y)):
        print(f'[RANK {rank}] ERROR: NaN/Inf encountered after scatter.')
        comm.Barrier()
        sys.exit(1)

    return local_X, local_y, counts, displs

# -------------------- training --------------------
def train_parallel(X: np.ndarray, y: np.ndarray, *,
                   hidden: int, activ: str, lr: float, batch: int, epoch: int, scaler: dict):
    m = X.shape[1]
    n = hidden
    N = X.shape[0]
    theta = init_params(m, n) if rank == 0 else None
    theta = comm.bcast(theta, root=0)
    losses = []

    for ep in range(1, epoch + 1):

        epoch_loss = 0.0
        for start in range(0, N, batch):
            end = min(start + batch, N)
            Xb, yb = X[start:end], y[start:end]
            grad, loss = gradient(Xb, yb, theta, m, n, activ)
            # Global gradients & global loss
            comm.Allreduce(MPI.IN_PLACE, grad, op=MPI.SUM)
            grad /= size
            loss = comm.allreduce(loss, op=MPI.SUM) / size
            theta -= lr * grad
            epoch_loss += loss * (end - start)
        epoch_loss /= N
        losses.append(epoch_loss)

        if rank == 0:
            print(f'Epoch {ep:3d}  R(θ) = {epoch_loss:.12f}')
        
    return theta, losses

# -------------------- main entry point --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--activ', choices=['relu', 'tanh', 'sigmoid'], default='relu')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # Process 0 is responsible for reading the data and scattering it
    if rank == 0:
        train_npz = np.load(os.path.join(args.data_dir, 'train_all.npz'))
        test_npz  = np.load(os.path.join(args.data_dir, 'test_all.npz'))
        scaler    = np.load(os.path.join(args.data_dir, 'scaler_all.npz'))
        X_train, y_train = train_npz['X'], train_npz['y']
        X_test,  y_test  = test_npz['X'],  test_npz['y']
        # Global shuffle
        perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]
        print(f"[DEBUG] y_train: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        print(f"[DEBUG] y_test:  min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}, std={y_test.std():.4f}")
        print(f'[INFO] Train set {X_train.shape} Test set {X_test.shape}')
    else:
        X_train = y_train = X_test = y_test = scaler = None


    # bcast scaler
    if rank == 0:
        scaler = dict(y_scale=scaler['y_scale'], y_min=scaler['y_min'])
    scaler = comm.bcast(scaler, root=0)
    # scatter train & test
    X_train, y_train, _, _ = scatter_data(X_train, y_train)
    X_test,  y_test,  _, _ = scatter_data(X_test,  y_test)

    t0 = time.time()
    theta, losses = train_parallel(X_train, y_train,
                                   hidden=args.hidden, activ=args.activ,
                                   lr=args.lr, batch=args.batch, epoch=args.epoch, scaler=scaler)
    if rank == 0:
        print(f'[INFO] Training time {time.time()-t0:.2f} s')

    # parallel RMSE
    train_rmse = parallel_rmse(X_train, y_train, theta,
                               X_train.shape[1], args.hidden, args.activ,
                               scaler['y_scale'], scaler['y_min'])
    test_rmse  = parallel_rmse(X_test,  y_test,  theta,
                               X_train.shape[1], args.hidden, args.activ,
                               scaler['y_scale'], scaler['y_min'])
    if rank == 0:
        print(f'[RESULT] Train RMSE = {train_rmse:.10f}')
        print(f'[RESULT] Test RMSE = {test_rmse:.10f}')
        plt.figure(figsize=(12, 8), dpi=300)
        
        # bold
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        
        # plot loss
        plt.plot(losses, linewidth=2.5, color='tab:blue', marker='o', markersize=3, markevery=1)
        plt.yscale('log')
        print(len(losses))
        
        plt.title(f'activation_function={args.activ} hidden={args.hidden} batch={args.batch} procs={size}', 
                fontsize=16)
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('R(θ)', fontsize=16)
        
        plt.xlim(-1, len(losses))
        if len(losses) > 20:
            step = max(1, len(losses) // 5)
            plt.xticks(range(0, len(losses), step))
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14, width=2)
        
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
        plt.tight_layout(pad=3.0)
        
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/mpi_loss_{args.activ}_h{args.hidden}_b{args.batch}_p{size}.png', 
                    bbox_inches='tight', dpi=300)
        print(f'[INFO] loss curves have been saved. → figures/mpi_loss_*.png')

if __name__ == '__main__':
    main()