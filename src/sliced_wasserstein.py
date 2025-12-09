import numpy as np
import torch

def sliced_wasserstein_2(X: torch.Tensor, Y: torch.Tensor, K: int = 64, device='cpu') -> float:
    n, d = X.shape
    
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).to(device)
    if not isinstance(Y, torch.Tensor):
        Y = torch.from_numpy(Y).to(device)
    
    X = X.float()
    Y = Y.float()

    V = torch.randn(K, d, device=device, dtype=torch.float32)
    norms = torch.linalg.norm(V, dim=1, keepdim=True)
    V = V / (norms + 1e-10)
    
    X_proj = X @ V.T  
    Y_proj = Y @ V.T  
    
    X_proj = torch.sort(X_proj, dim=0).values
    Y_proj = torch.sort(Y_proj, dim=0).values
    
    w_2 = torch.mean((X_proj - Y_proj) ** 2)
    
    return float(w_2)


def sliced_wasserstein_2_barycenter(Y_list: list[torch.Tensor], 
                                        rhos: torch.Tensor, 
                                        K: int = 32, 
                                        step_size: float = 1.0, 
                                        n_iter: int = 100,
                                        device='cpu') -> torch.Tensor:
    J = len(Y_list)
    n, d = Y_list[0].shape
    
    # Ensure all on GPU
    Y_list = [y.to(device).float() if not y.is_cuda else y.float() for y in Y_list]
    rhos = rhos.to(device).float() if not rhos.is_cuda else rhos.float()
    
    # Initialize with random sample from first distribution
    idx = np.random.randint(0, n)
    X = Y_list[0][idx:idx+1].repeat(n, 1)
    
    for _ in range(n_iter):
        gradient = torch.zeros((n, d), dtype=torch.float32, device=device)
        H = torch.zeros((d, d), dtype=torch.float32, device=device)
        
        V = torch.randn(K, d, device=device, dtype=torch.float32)
        norms = torch.linalg.norm(V, dim=1, keepdim=True)
        mask = norms.squeeze() > 0
        V = V[mask]
        V = V / torch.linalg.norm(V, dim=1, keepdim=True)
        K_actual = len(V)
        
        if K_actual == 0:
            continue
        
        for k in range(K_actual):
            theta = V[k]
            
            X_proj = X @ theta
            sort_idx = torch.argsort(X_proj)
            X_proj_sorted = X_proj[sort_idx]
            
            diff_sorted = torch.zeros(n, dtype=torch.float32, device=device)
            for j in range(J):
                Y_proj_j = torch.sort(Y_list[j] @ theta).values
                diff_sorted += rhos[j] * (X_proj_sorted - Y_proj_j)
            
            diff = torch.zeros(n, dtype=torch.float32, device=device)
            diff[sort_idx] = diff_sorted
            
            gradient += torch.outer(diff, theta)
            H += torch.outer(theta, theta)
        
        gradient /= K_actual
        H /= K_actual
        
        H_pinv = torch.linalg.pinv(H)
        X -= step_size * (gradient @ H_pinv)
    
    return X

def sliced_wasserstein_2_projection(X: torch.Tensor,
                                        Y: torch.Tensor,
                                        K: int = 32,
                                        step_size: float = 1.0,
                                        n_iter: int = 100,
                                        device='cpu') -> torch.Tensor:
    n, d = X.shape
    X = X.clone()
    
    # Ensure on GPU
    X = X.to(device).float()
    Y = Y.to(device).float()
    
    for _ in range(n_iter):
        gradient = torch.zeros((n, d), dtype=torch.float32, device=device)
        H = torch.zeros((d, d), dtype=torch.float32, device=device)
        
        V = torch.randn(K, d, device=device, dtype=torch.float32)
        norms = torch.linalg.norm(V, dim=1, keepdim=True)
        mask = (norms.squeeze() > 1e-10)
        V = V[mask]
        if len(V) == 0:
            continue
        V = V / torch.linalg.norm(V, dim=1, keepdim=True)
        K_actual = len(V)
        
        for k in range(K_actual):
            theta = V[k]
            
            X_proj = X @ theta
            sort_idx = torch.argsort(X_proj)
            X_proj_sorted = X_proj[sort_idx]
            
            Y_proj = torch.sort(Y @ theta).values
            diff_sorted = X_proj_sorted - Y_proj
            
            diff = torch.zeros(n, dtype=torch.float32, device=device)
            diff[sort_idx] = diff_sorted
            
            gradient += torch.outer(diff, theta)
            H += torch.outer(theta, theta)
        
        gradient /= K_actual
        H /= K_actual
        
        H_pinv = torch.linalg.pinv(H)
        X -= step_size * (gradient @ H_pinv)
    
    return X