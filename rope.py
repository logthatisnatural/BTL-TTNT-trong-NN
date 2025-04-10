from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    # Step 1: Compute the frequencies
    # Create position indices [0, 1, ..., seqlen-1]
    positions = torch.arange(seqlen, device=device)
    # Compute the frequencies: freq = position / (theta^(2i/head_dim))
    freq_indices = torch.arange(0, head_dim, 2, device=device).float()  # [0, 2, 4, ..., head_dim-2]
    freqs = theta ** (-freq_indices / head_dim)  # [theta^(-0/head_dim), theta^(-2/head_dim), ...]
    freqs = positions[:, None] * freqs[None, :]  # Shape: (seqlen, head_dim//2)

    # Step 2: Compute cos and sin for the angles
    cos = torch.cos(freqs)  # Shape: (seqlen, head_dim//2)
    sin = torch.sin(freqs)  # Shape: (seqlen, head_dim//2)

    # Step 3: Reshape query and key to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # query_real: q_1, q_3, q_5, ... (shape: (batch_size, seqlen, n_local_heads, head_dim//2))
    # query_imag: q_2, q_4, q_6, ... (shape: (batch_size, seqlen, n_local_heads, head_dim//2))

    # Step 4: Reshape cos and sin for broadcasting
    cos = reshape_for_broadcast(cos, query_real)  # Shape: (1, seqlen, 1, head_dim//2)
    sin = reshape_for_broadcast(sin, query_real)  # Shape: (1, seqlen, 1, head_dim//2)

    # Step 5: Apply rotary transformation
    # q'_i = q_i * cos - q_(i+1) * sin
    # q'_(i+1) = q_i * sin + q_(i+1) * cos
    query_out_real = query_real * cos - query_imag * sin
    query_out_imag = query_real * sin + query_imag * cos
    key_out_real = key_real * cos - key_imag * sin
    key_out_imag = key_real * sin + key_imag * cos

    # Step 6: Combine real and imaginary parts back into the original shape
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).reshape(query.shape)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).reshape(key.shape)

    # Step 7: Return the rotary position embeddings for the query and key tensors
    return query_out, key_out