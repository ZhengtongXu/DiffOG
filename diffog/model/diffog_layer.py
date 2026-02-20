import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from copy import deepcopy
from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable
from torch.nn.parameter import Parameter as Parame

class DiffogTf(nn.Module):
    """
    Differentiable trajectory optimization layer using a Transformer encoder.
    
    Takes an input action trajectory and solves a Quadratic Program (QP) to produce
    an optimized trajectory that satisfies smoothness and constraint requirements.
    
    Two forward modes:
        - forward(): enforces inter-step smoothness constraints.
        - forward_with_past_action(): additionally constrains the first action to be
          close to the last executed action, ensuring temporal consistency at inference.
    
    Args:
        pred_horizon (int): Number of predicted time steps.
        single_ac_dim (int): Dimension of a single action.
        device (str): Device for computation.
        direction (str): 'minus' or 'plus', sign convention for the QP linear term.
        constraint (float): Bound on consecutive action differences.
        embed_dim (int): Transformer embedding dimension.
        num_heads (int): Number of attention heads in the Transformer encoder.
        num_layers (int): Number of Transformer encoder layers.
        dim_feedforward (int): Feedforward dimension in Transformer layers.
        dropout (float): Dropout rate for the Transformer encoder.
        pooling (str): Pooling strategy over time ('mean', 'max', 'cls', 'attention').
        smooth_weight (float): Weight for the smoothness regularization term.
    """
    def __init__(
        self,
        pred_horizon=16,
        single_ac_dim=7,
        device='cuda:0',
        direction='minus',
        constraint=0.1,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        pooling='mean', # Options: 'mean', 'max', 'cls'
        smooth_weight=0.0,
    ):
        super().__init__()
        self.device = device

        self.encoder = TemporalTransformerEncoder(
            input_dim=single_ac_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling  # Options: 'mean', 'max', 'cls'
        )
        self.smooth_weight = smooth_weight
        print('smooth_weight',smooth_weight)
        self.single_ac_dim = single_ac_dim
        act_dim = single_ac_dim
        n_pred = pred_horizon

        self.L_linear = nn.Linear(embed_dim, (act_dim*n_pred)*(act_dim*n_pred)).to(self.device)
        self.norm = nn.LayerNorm(embed_dim).to(self.device)
        self.constraint = constraint
        self.eps = 1e-4
        self.direction = direction
        self.output_dim = act_dim*n_pred

        sub_diag = torch.eye(act_dim, requires_grad=False)
        sub_diff = torch.cat((-sub_diag, sub_diag), 1)

        A_sat = torch.cat((sub_diff, torch.zeros(act_dim, act_dim*n_pred - act_dim*2, requires_grad=False)), 1)
        for i in range(1, n_pred-1):
            row = torch.cat((torch.zeros(act_dim, act_dim*i, requires_grad=False), sub_diff), 1)
            row = torch.cat((row, torch.zeros(act_dim, act_dim*n_pred - act_dim*(2+i), requires_grad=False)), 1)
            A_sat = torch.cat((A_sat, row), 0)

        if act_dim == 7:
            for i in range(n_pred-1):
                A_sat = A_sat[torch.arange(A_sat.size(0)) != (n_pred-i-1)*act_dim - 1]
            self.b_sat_single = torch.tensor([constraint,constraint,constraint,constraint,constraint,constraint], requires_grad=False).reshape(act_dim-1,1)

        else:
            assert (act_dim == 7)

        self.A_sat_single = A_sat
        self.A_sat = torch.cat((A_sat, -A_sat), 0)

        self.b_sat = self.b_sat_single
        for i in range(2*(n_pred-1)-1):
            self.b_sat = torch.cat((self.b_sat, self.b_sat_single), 0)

        if act_dim == 7:
            A_single = torch.eye(act_dim-1, requires_grad=False)
            A_single_zero = torch.cat((A_single, torch.zeros(act_dim-1, act_dim*n_pred - act_dim+1, requires_grad=False)), 1)
            self.A_past_action = torch.cat((A_single_zero, -A_single_zero), 0)
        else:
            assert (act_dim == 7)

    def forward(self, x):

        assert x.ndimension() == 3

        n_batch, single_horizon, single_act_dim = x.shape
        act_dim = single_horizon * single_act_dim

        tf_feat = self.encoder(x)
        x = x.reshape(n_batch, act_dim)

        tf_feat = self.norm(tf_feat)

        L_unconstrained = self.L_linear(tf_feat) 
        L_unconstrained = L_unconstrained.view(n_batch, act_dim, act_dim)

        diag_mask = torch.eye(act_dim, device=x.device, dtype=torch.bool).unsqueeze(0).expand(n_batch, act_dim, act_dim)

        L_unconstrained = L_unconstrained.clone()
        diag_vals = torch.exp(L_unconstrained[diag_mask]) + self.eps
        L_unconstrained[diag_mask] = diag_vals
        L_unconstrained = torch.clamp(L_unconstrained, min=-10.0, max=10.0)
        L = torch.tril(L_unconstrained)
        Q = torch.matmul(L, L.transpose(-1, -2)) + self.eps * torch.eye(self.output_dim, requires_grad=False, device=x.device).unsqueeze(0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        e = Variable(torch.Tensor())

        diff_mat = self.smooth_weight*(self.A_sat_single.t().mm(self.A_sat_single)).to(x.device)

        diff_mat = diff_mat.unsqueeze(0).expand(n_batch, act_dim, act_dim).to(x.device)

        # make diff_mat not require grad
        diff_mat = diff_mat.detach()

        if self.smooth_weight > 0:
            Q = Q + diff_mat


        if self.direction == 'plus':
            inputs = x
        elif self.direction == 'minus':
            inputs = -x
        else:
            raise ValueError("Invalid direction")

        A_sat_batch = self.A_sat.unsqueeze(0).expand(n_batch, self.A_sat.shape[0], self.A_sat.shape[1]).to(x.device)
        b_sat_batch = self.b_sat.unsqueeze(0).expand(n_batch, self.b_sat.shape[0], 1).to(x.device).reshape(n_batch, self.b_sat.shape[0])

        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), A_sat_batch.double(), b_sat_batch.double(), e, e
        )
        x = x.float()

        mean = x.view(n_batch, single_horizon, single_act_dim)

        return mean
    
    def forward_with_past_action(self, x, past_action):

        assert x.ndimension() == 3

        n_batch, single_horizon, single_act_dim = x.shape
        act_dim = single_horizon * single_act_dim
  
        tf_feat = self.encoder(x)
        x = x.reshape(n_batch, act_dim)

        tf_feat = self.norm(tf_feat)

        L_unconstrained = self.L_linear(tf_feat) 
        L_unconstrained = L_unconstrained.view(n_batch, act_dim, act_dim)

        diag_mask = torch.eye(act_dim, device=x.device, dtype=torch.bool).unsqueeze(0).expand(n_batch, act_dim, act_dim)

        L_unconstrained = L_unconstrained.clone()
        diag_vals = torch.exp(L_unconstrained[diag_mask]) + self.eps
        L_unconstrained[diag_mask] = diag_vals
        L_unconstrained = torch.clamp(L_unconstrained, min=-10.0, max=10.0)
        L = torch.tril(L_unconstrained)
        Q = torch.matmul(L, L.transpose(-1, -2)) + self.eps * torch.eye(self.output_dim, requires_grad=False, device=x.device).unsqueeze(0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        e = Variable(torch.Tensor())


        diff_mat = self.smooth_weight*(self.A_sat_single.t().mm(self.A_sat_single)).to(x.device)

        diff_mat = diff_mat.unsqueeze(0).expand(n_batch, act_dim, act_dim).to(x.device)

        # make diff_mat not require grad
        diff_mat = diff_mat.detach()

        if self.smooth_weight > 0:
            Q = Q + diff_mat


        if self.direction == 'plus':
            inputs = x
        elif self.direction == 'minus':
            inputs = -x
        else:
            raise ValueError("Invalid direction")

        A_sat_batch = self.A_sat.unsqueeze(0).expand(n_batch, self.A_sat.shape[0], self.A_sat.shape[1]).to(inputs.device)
        b_sat_batch = self.b_sat.unsqueeze(0).expand(n_batch, self.b_sat.shape[0], 1).to(inputs.device).reshape(n_batch, self.b_sat.shape[0])

        A_past_action_batch = self.A_past_action.unsqueeze(0).expand(n_batch, self.A_past_action.shape[0], self.output_dim).to(inputs.device)


        b_past_action_batch_upper = past_action[:,:,0:6].reshape(n_batch, 6) + self.constraint*torch.ones(n_batch, 6,requires_grad = False).to(inputs.device) 
        b_past_action_batch_lower = -1*past_action[:,:,0:6].reshape(n_batch, 6) + self.constraint*torch.ones(n_batch, 6,requires_grad = False).to(inputs.device)

        b_past_action_batch = torch.cat((b_past_action_batch_upper,b_past_action_batch_lower),1)   

        G = torch.cat((A_sat_batch,A_past_action_batch),1)
        h = torch.cat((b_sat_batch,b_past_action_batch),1)

        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float()

        mean = x.view(n_batch, single_horizon, single_act_dim)

        return mean

class TemporalTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for time series data with a single embedding output.
    """
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dim_feedforward=256, dropout=0.0, pooling='mean'):
        """
        Args:
            input_dim (int): The dimension of each feature (D).
            embed_dim (int): The embedding dimension for the transformer.
            num_heads (int): The number of attention heads.
            num_layers (int): Number of transformer layers.
            dim_feedforward (int): Dimension of feedforward network in each layer.
            dropout (float): Dropout rate.
            pooling (str): Pooling method to aggregate over time. Options: ['mean', 'max', 'cls'].
        """
        super(TemporalTransformerEncoder, self).__init__()
        
        # Input linear projection to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder layers
        print('dropout',dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch-first ordering (B, T, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling method
        self.pooling = pooling

        if self.pooling == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, 1)
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, embed_dim).
        """
        B, T, D = x.shape
        
        # Project input to embedding dimension
        x = self.input_proj(x)

        # Transformer encoder
        x = self.transformer(x)

        # Pooling over time dimension
        if self.pooling == 'mean':
            # Mean pooling
            x = x.mean(dim=1)  # Shape (B, embed_dim)
        elif self.pooling == 'max':
            # Max pooling
            x, _ = x.max(dim=1)  # Shape (B, embed_dim)
        elif self.pooling == 'cls':
            x = x[:, 0, :]  # Shape (B, embed_dim)
        elif self.pooling == 'attention':
            # Attention pooling
            attn_scores = self.attention_weights(x)  # Shape (B, T, 1)
            attn_scores = torch.softmax(attn_scores, dim=1)  # Shape (B, T, 1)
            x = (x * attn_scores).sum(dim=1)  # Shape (B, embed_dim)
        return x
    
