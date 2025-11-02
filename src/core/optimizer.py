import torch
from torch import Tensor


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Adapted from nanochat implementation.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Simplified version for single GPU usage.
    Should not be used for embedding layers or final fully connected layers.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        
        # Group parameters by size for efficiency
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                # Check for NaN/Inf in gradients
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"WARNING: Skipping parameter update due to NaN/Inf gradient")
                    continue
                
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Apply Newton-Schulz orthogonalization
                try:
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                except Exception as e:
                    print(f"WARNING: NS5 failed, using gradient directly: {e}")
                    # Fall back to regular gradient if NS5 fails
                    pass
                
                # Check again after NS5
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"WARNING: NaN/Inf after NS5, skipping update")
                    continue
                
                # Apply aspect-ratio scaled step
                scale = max(1, p.size(-2) / p.size(-1))**0.5
                p.add_(g, alpha=-group["lr"] * scale)
        
        return loss
