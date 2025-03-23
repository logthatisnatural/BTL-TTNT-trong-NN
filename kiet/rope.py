from typing import Tuple
import torch

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    """
    batch_size, seqlen, num_heads, _ = query.shape
    device = query.device

    # Tạo index vị trí token (0, 1, ..., seqlen-1)
    position_ids = torch.arange(seqlen, dtype=torch.float32, device=device).unsqueeze(1)  # (seqlen, 1)

    # Tạo giá trị theta cho mỗi chiều của vector
    theta_inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))

    # Nhân với index vị trí để tạo embeddings (seqlen, head_dim/2)
    position_embeddings = position_ids * theta_inv.unsqueeze(0)  # (seqlen, head_dim/2)

    # Tính toán sin/cos
    sin_emb = torch.sin(position_embeddings)  # (seqlen, head_dim/2)
    cos_emb = torch.cos(position_embeddings)  # (seqlen, head_dim/2)

    # Tách query và key thành phần real và imag
    query_real, query_imag = query.float().reshape(batch_size, seqlen, num_heads, -1, 2).unbind(-1)
    key_real, key_imag = key.float().reshape(batch_size, seqlen, num_heads, -1, 2).unbind(-1)

    # Đảm bảo sin/cos có cùng shape với query và key
    sin_emb = sin_emb.view(1, seqlen, 1, -1)  # (1, seqlen, 1, head_dim/2)
    cos_emb = cos_emb.view(1, seqlen, 1, -1)  # (1, seqlen, 1, head_dim/2)

    # Áp dụng RoPE
    query_out_real = query_real * cos_emb - query_imag * sin_emb
    query_out_imag = query_imag * cos_emb + query_real * sin_emb
    query_out = torch.cat([query_out_imag, query_out_real], dim=-1)  # Đảo lại thứ tự

    key_out_real = key_real * cos_emb - key_imag * sin_emb
    key_out_imag = key_imag * cos_emb + key_real * sin_emb
    key_out = torch.cat([key_out_imag, key_out_real], dim=-1)  # Đảo lại thứ tự

    return query_out, key_out
