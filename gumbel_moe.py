import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, b=False):
        super().__init__()
        # In Linear layer in_dim equal token embedding dimensions = number of features
        # hidden_dim is wanted number of features after linear transformation
        # Linear 1 equals (Wx + b)
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=b)
        # Linear 2 equals (Vx + c)
        self.linear2 = nn.Linear(in_dim, hidden_dim, bias=b)
        # Linear 3 equals W2
        self.linear3 = nn.Linear(hidden_dim, in_dim, bias=b)
        # beta is used to create Swish
        if b:
            self.beta = nn.Parameter(torch.randn(1 , hidden_dim, requires_grad = True))
        else:
            self.beta = nn.Parameter(torch.ones(1 , hidden_dim))
        self.glu = nn.GLU()
        
    def forward(self, x :torch.Tensor):
        # equals to SILU, but instead of x * sigmoid(x) --> x * sigmoid( x * beta)
        glu_input = torch.cat([self.linear1(x), self.linear1(x) * self.beta], dim = x.ndim -1)
        swish = self.glu(glu_input)
        linear2 = self.linear2(x)
        return self.linear3(linear2 * swish)


class GEGLU(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, b=False):
        super().__init__()
        # In Linear layer in_dim equal token embedding dimensions = number of features
        # hidden_dim is wanted number of features after linear transformation
        # Linear 1 equals (Wx + b)
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=b)
        # Linear 2 equals (Vx + c)
        self.linear2 = nn.Linear(in_dim, hidden_dim, bias=b)
        # Linear 3 equals W2
        self.linear3 = nn.Linear(hidden_dim, in_dim, bias=b)
        self.gelu = nn.GELU()
        
    def forward(self, x :torch.Tensor):
        linear1 = self.linear1(x)
        linear2 = self.linear2(x)
        gelu = self.gelu(linear1)
        return self.linear3(gelu * linear2)
    

class RMS_norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, b=False):
        super().__init__()
        self.eps = eps
        # Equals to gamma = 1
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _normalize(self, x: torch.Tensor):
        # torch.rsqrt = 1 / sqrt(input i)
        # i = sum(x^2) / n = mean(x^2) 
        # -1 make sure that mean is calculated using each tensors rows
        return x * torch.rsqrt( x.pow(2).mean(-1, keepdim=True) + self.eps )
    
    def forward(self, x: torch.Tensor):
        # dim : (B, seq_len, emb_dim) -> (B, seq_len, emb_dim)
        return self.weight * self._normalize(x.float()).type_as(x)

class Layer_norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, b=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=b, bias=b)
        
    def forward(self, x: torch.Tensor):
        return self.layer_norm(x)
    

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, b=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=b)
        
    def forward(self, x:torch.Tensor):
        x = self.linear(x)
        return x


def precompute_theta_pos_frequencies(emb_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert emb_dim % 2 == 0, "Dimension must be divisible by 2"
    # Creating theta parameter, theta_i = 10000 ^( -2 * (i -1) / emb_dim) for i = [1,2, ..., emb_dim / 2]
    # Theta numerator = (i -1) for i in range [1,2, ..., emb_dim / 2]
    theta_numerator = torch.arange(0, emb_dim, 2, device=device).float()
    # x^(-2) = 1 / x^2, x^(-2 * (2-1) ) = 1 / x^2, x^(-2 * (3-1) ) = 1 / x^4, x^(-2 * (4-1) ) = 1 / x^6
    theta_values = 1.0 / (theta ** (theta_numerator / emb_dim))
    # m = token position, generating every possible m between 0 and seq_len - 1 (maxium lenght of sequence)
    m = torch.arange(seq_len, device=device)
    # to calculate cos and sin (m * θ _1 ) , ... , cos and sin (m * θ _(emb_dim / 2)) for every m
    # torch outer can be used --> it calculates every combination between two vectors
    # m[0] * theta[0], .... , m[0] * theta[emb_dim / 2]
    # .
    # .
    # m[seq_len] * theta[0], .... , m[seq_len] * theta[emb_dim / 2]
    freqs_matrix = torch.outer(m, theta_values).float()
    # freqs neef to be converted to coplex numbers, so that new matrix is  
    # cos(m[0] * theta[0]) + i*sin(m[0] * theta[0]) , .... , cos(m[0] * theta[emb_dim / 2]) + i*sin(m[0] * theta[emb_dim/ 2])
    # .
    # .
    # cos(m[seq_len] * theta[0]) + i*sin(m[seq_len] * theta[0]) , .... , etc.
    freqs_complex = torch.polar(torch.ones_like(freqs_matrix, device=device), freqs_matrix)
    # ------
    return freqs_complex

def RoPE(x: torch.Tensor, freqs: torch.Tensor, index: int = -1):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs.shape[-1] == x.shape[-1]
    
    if index != -1:
        # Ensure that index cannot be less than 1 --> (1 == first token position) 
        # and more than freqs.shape[-2] --> (index > max sequence length)
        index = max(1, min(index, freqs.shape[-2]))
        freqs = freqs[(index -1), :].unsqueeze(0)
        
    if freqs.shape[-2] != x.shape[-2] and index == -1:
        # if in x seq_len != max lenght, drop unneeded m frequence positions 
        freqs = freqs[:x.shape[-2], :]
    
    # from (seq_len, head_dim) to (1, 1, seq_len, head_dim)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    shape = [d if i > 1 else 1 for i, d in enumerate(x.shape)]
    # Reshape frequence tensor from (1, 1, seq_len, head_dim) --> (1, 1, seq_len, pairs_count, 2)
    # this allow broadcasting when batch size is more than 1
    freqs = freqs.view(*shape)
    rotated_x = x * freqs
     
    return rotated_x

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # In tensor row level [x_1, x_2 , ..., x_d-1, x_d], where d = head_dim
    # new shape row --> [ [x_1, x_2], [x_3, x_4], ... , [x_d-1, x_d] ]
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    # converts every row value pair [ [x_1, x_2], [x_3, x_4], ... , [x_d-1, x_d] ]
    # to [ [x_1 + x_2 * i], [x_3 + x_4 * i] ... etc. ] complex form    
    x_complex = torch.view_as_complex(x_reshaped.float())
    x_rotated = RoPE(x_complex, freqs_complex)
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out


# --- Transformer --- #

class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
                
        self.attention_type = args.attention_type
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        
        # Regardles of Attention type Qw is always same size
        self.Q = Linear(args.dim, args.dim)
        
        if args.attention_type == 'MultiH':
            self.K = Linear(args.dim, args.dim)
            self.V = Linear(args.dim, args.dim)
            
        elif args.attention_type == 'MultiQ':            
            self.K = Linear(args.dim, args.head_dim)
            self.V = Linear(args.dim, args.head_dim)
            
    def forward(self, x : torch.Tensor, mask: Optional[torch.Tensor], freqs: torch.Tensor):
            
        B, seq_len, _ = x.shape
        attention_type = self.attention_type
            
        xq = self.Q(x)
        xk = self.K(x)
        xv = self.V(x)
            
        # Reshaping xQ, xV and xK
            
        if attention_type == 'MultiH':
            xk = xk.reshape(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            xv = xv.reshape(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            xq = xq.reshape(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            
        elif attention_type == 'MultiQ':
            # in case of Multi-Query Attention xV and xK is reshaped from (B,seq_len, head_dim)
            # to (B, 1, seq_len, head_dim)
            xk = xk.unsqueeze(1)
            xv = xv.unsqueeze(1)
            # reshaping xV and xK again to shape (B, n_heads, seq_len, head_dim)
            xk = xk.expand(-1, self.n_heads, -1, -1)
            xv = xv.expand(-1, self.n_heads, -1, -1)
            # reshaping xQ to correspond shape of xK and xV
            xq = xq.reshape(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                 
        # Rotating K and Q embeddings
        freqs = freqs[self.head_dim]
        xq = apply_rotary_embeddings(xq, freqs)
        xk = apply_rotary_embeddings(xk, freqs)
        
            
        # all xQ, xV and xK are currently shape (B, n_groups, seq_len, head_dim) or (B, n_heads, seq_len, head_dim)
        # to calcul Q*K^T, xQ needs to be transposed
        xk = xk.transpose(-2, -1)
        # calculating softmax(Q*K^T / sqrt(emd_dim)) i.e scores, shape (B, n_heads or n_groups, seq_len, seq_len)
        scores = torch.matmul(xq, xk) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Broadcasting Mask shape equal to score shape
            # Needed to mask padding tokens
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, scores.size(1), -1, -1) 
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # shape (B, n_heads or n_groups, seq_len, head_dim)
        attention = torch.matmul(scores, xv)
        # After transpose shape (B, seq_len, n_heads or n_groups, head_dim), then reshaped to (B, seq_len, emb_dim)
        attention = attention.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return attention
    
class Layer_Block(nn.Module):
    def __init__(self, args, dropout=0.0):
        super().__init__()
        
        self.dim = args.dim
        self.ff_type = args.ff_type
        self.norm_1 = RMS_norm(args.dim) if args.norm == 'RMS' else Layer_norm(args.dim)
        self.norm_2 = RMS_norm(args.dim) if args.norm == 'RMS' else Layer_norm(args.dim)
        self.attention = SelfAttention(args)
        self.feedforward = SwiGLU(args.dim, args.dim * 4) if args.ff_type == 'SwiGLU' else GEGLU(args.dim, args.dim * 4)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x : torch.Tensor, mask: Optional[torch.Tensor], freqs: torch.Tensor):
        # RMS_norm (X)
        x_norm = self.norm_1(x)
        # Attention (normalized X)
        A_x = self.attention(x_norm, mask, freqs)
        # Multihead Attention result + X
        A_x = self.dropout1(A_x) + x
        # RMS_norm (Attention result + X) 
        A_x_norm = self.norm_2(A_x)
        # SwiGLU or GEGLU ( A_x_norm )
        x_ff = self.feedforward(A_x_norm)
        x_ff = self.dropout2(x_ff)
        x_out =  A_x + x_ff
        return x_out

class ParallelLayer(nn.Module):
    def __init__(self, args, out_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([Layer_Block(args) for _ in range(args.n_parallel_blocks)])
        self.out_dim = out_dim
        if out_dim is not None:
            self.out_linear = Linear(args.dim, out_dim)
        self.MoE = GumbelMoE(args.dim* 2, args.dim)
        
    def forward(self, x, mask, freqs):
        outputs = [block(x, mask, freqs) for block in self.blocks]
        output, entropy_loss, load_balance_loss = self.MoE(outputs[0], outputs[1])
        if self.out_dim is not None:
           output = self.out_linear(output)
        return output, entropy_loss, load_balance_loss

class LargeLayer(nn.Module):
    def __init__(self, args, out_dim=None):
        super().__init__()
        self.block = Layer_Block(args)
        self.out_dim = out_dim
        if self.out_dim is not None:
            self.Share = Linear(args.dim, self.out_dim)
            
    def forward(self, x, mask, freqs):
        output = self.block(x, mask, freqs)
        if self.out_dim is not None:
            output = self.Share(output)
        return output


class GumbelMoE(nn.Module):
    def __init__(self, dim_in, dim_out, temperature=1.0):
        super().__init__()
        self.dim_in = dim_in          # path_dim * 2
        self.dim_path = dim_out       # path_dim

        self.share_linear = nn.Linear(dim_in, dim_out)     # maps concat(attn1, attn2) to x_s
        self.router = nn.Linear(dim_out, 3)                # maps logits to choose path

        self.temperature = temperature

    def gumbel_softmax(self, logits, temperature, hard=False):
        eps = 1e-10
        U = torch.rand_like(logits).clamp(min=eps, max=1.0)
        gumbels = -torch.log(-torch.log(U))
        y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)
    
        if hard:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter(-1, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        else:
            return y_soft

    def forward(self, attn1, attn2):
        # attn1, attn2 --> (B, seq_len, path_dim)
        combined_x = torch.cat([attn1, attn2], dim=-1)      # (B, seq_len, 2 * path_dim)
        x_s = self.share_linear(combined_x)                 # (B, seq_len, path_dim)

        router_logits = self.router(x_s)                    # (B, seq_len, 3)
        router_probs = self.gumbel_softmax(router_logits, self.temperature, hard=True)
    
        router_probs_expanded = router_probs.unsqueeze(-1)
        candidates = torch.stack([attn1, attn2, x_s], dim=2)
        output = torch.sum(router_probs_expanded * candidates, dim=2)
        
        # Entropy regularization
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1)  # (B, seq_len)
        entropy_loss = -torch.mean(entropy)  # scalar

        # Load balancing
        mean_probs = router_probs.mean(dim=(0, 1))  # shape (3,)
        load_balance_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))  # scalar

        return output, entropy_loss, load_balance_loss
       

            
class Decoder(nn.Module):
    def __init__(self, args, args_paraller):
        super().__init__()
        self.device = args.device
        self.args = args
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.n_layers = args.n_layers
        self.embeddings = nn.Embedding(self.vocab_size, args.dim, padding_idx=self.pad_id)
        self.freqs = {
            args.head_dim: precompute_theta_pos_frequencies(args.head_dim, args.max_seq_len * 2, device=self.device),
            args_paraller.head_dim: precompute_theta_pos_frequencies(args_paraller.head_dim, args.max_seq_len * 2, device=self.device)
        }
        
        self.layers = nn.ModuleList()
        
        # First Large Block
        self.layers.append(LargeLayer(args, args_paraller.dim))
        # Paraller Blocks
        for idx in range(args.n_layers):
            self.layers.append(ParallelLayer(args_paraller))
        # Last Paraller Block needs Grow linear
        self.layers.append(ParallelLayer(args_paraller, args.dim))
        # Second Large Block
        self.layers.append(LargeLayer(args))
    
        self.norm = RMS_norm(args.dim) if args.norm == 'RMS' else Layer_norm(args.dim)
        self.Linear = Linear(args.dim, args.vocab_size)
        
    def forward(self, x : torch.Tensor, classification=False):
        B, seqlen = x.shape
    
        # intialing padding mask
        padding_mask = (x == self.pad_id).unsqueeze(1).expand(-1, seqlen, -1)
        inf_mask = torch.full(padding_mask.shape, float('-inf'), device=self.device)
        zero_mask = torch.zeros(padding_mask.shape, device=self.device)
        padding_mask = torch.where(padding_mask, inf_mask, zero_mask)
        # intialing normal mask
        normal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=self.device), diagonal=1)
        # compined mask
        mask = padding_mask + normal_mask
        
        # convertin token idx to embedding vectors
        x = self.embeddings(x)
                
        entropy_total = 0
        load_total = 0

        for layer in self.layers:
            if isinstance(layer, ParallelLayer):
                x, entropy_loss, load_loss = layer(x, mask, self.freqs)
                entropy_total += entropy_loss
                load_total += load_loss
            else:
                x = layer(x, mask, self.freqs)
            
        x_norm = self.norm(x)
        x_out = self.Linear(x_norm).float()
        
        return x_out, entropy_total, load_total
