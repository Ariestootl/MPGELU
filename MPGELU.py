
import torch as torch
import math

class MPGELU(torch.nn.Module):
    '''
    Modified Parametric Gaussian Error Linear Unit (MPGELU) Activation Function
    '''
    def __init__(self,initial_s: float = 0.0, use_softplus: bool = True):
        super(MPGELU, self).__init__()
        self.initial_s = torch.nn.Parameter(torch.tensor(initial_s, dtype=torch.float32))
        self.use_softplus = use_softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_softplus:
            lam = 1.0 + torch.nn.functional.softplus(self.initial_s)
        else:
            lam = 1.0 + torch.log(torch.tensor(1.0, device=x.device) + torch.exp(self.initial_s))
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(lam,x)/ (math.sqrt(2)))))
        return output
        

