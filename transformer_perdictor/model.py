from __future__ import annotations

import copy
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a positional encoding matrix of shape [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer ensures it's saved in the state_dict but not trained as a parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1), :]

class USDCNHTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        seq_len,
        error_history_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        patch_len=8,
        patch_stride=4,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.patch_len = min(max(2, patch_len), seq_len)
        self.patch_stride = max(1, patch_stride)

        n_patches = 1 + max(0, (seq_len - self.patch_len) // self.patch_stride)
        self.n_patches = n_patches
        
        # use 1D conv before projection to capture local temporal patterns within patches (optional, can be removed for simplicity)
        self.conv1d = nn.Conv1d(input_size, d_model, kernel_size=3, padding=1) if patch_len > 2 else None
        # Patch embedding over time: [B, N, patch_len * features] -> [B, N, d_model]
        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_len * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Error State Projection: [error_history_len] -> [d_model]
        self.error_projection = nn.Sequential(
            nn.Linear(error_history_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for [error token + patch tokens]
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_patches + 1)

        # 2. Transformer Encoder (Standard)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2, # Often 4x is used (256), but 2x is fine for small datasets
            dropout=dropout,
            batch_first=True,
            activation='gelu', # GELU yields smoother gradients than ReLU
            norm_first=True
        )
        # Keep pre-norm stability (norm_first=True) and explicitly disable
        # nested tensor optimization to avoid PyTorch runtime warnings.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 3. Wider Shared Representation
        # Expand from d_model to a larger shared feature space.
        shared_dim = d_model * 2
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cond_1h = nn.Linear(2, d_model * 2) # Maps (mu, sigma) to shared_dim
        self.cond_4h = nn.Linear(2, d_model * 2) # Maps (mu, sigma) to shared_dim

        # 4. Dedicated Multi-Task Heads
        head_hidden = max(1, shared_dim // 2)
        self.head_15m = self._build_head(shared_dim, head_hidden, out_dim=2, dropout=dropout)
        self.head_1h = self._build_head(shared_dim, head_hidden, out_dim=2, dropout=dropout)
        self.head_4h = self._build_head(shared_dim, head_hidden, out_dim=2, dropout=dropout)

    def _build_head(self, in_dim, hidden_dim, out_dim, dropout):
        """Helper to build a dedicated 2-block MLP head for a specific task."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def _build_patches(self, x):
        """
        Args:
            x: [B, L, C] where C is input_size
        Returns:
            patches: [B, N, d_model] (after projection)
            padding_mask: [B, N]
        """
        b, l, c = x.shape
        original_len = l

        # 1. Apply Conv1D within the patching logic
        # We transpose to [B, C, L] because Conv1D expects channels in the second dim
        if self.conv1d is not None:
            x_t = x.transpose(1, 2) 
            x_t = self.conv1d(x_t)  # Output: [B, d_model, L]
            c = x_t.shape[1]        # Update C to d_model
        else:
            x_t = x.transpose(1, 2)

        # 2. Padding logic (ensuring sequence is long enough for at least one patch)
        if l < self.patch_len:
            x_t = F.pad(x_t, (0, self.patch_len - l), mode="constant", value=0.0)
            l = self.patch_len
                
        # 3. Create Patches using unfold
        # Resulting shape: [B, C, N_patches, patch_len]
        patches = x_t.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        
        # 4. Reshape to [B, N_patches, C * patch_len]
        # We permute to bring N_patches to the second dimension
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(b, patches.shape[1], -1) 

        # 5. Handle fixed N_patches dimension (consistency for Transformer)
        if patches.shape[1] != self.n_patches:
            if patches.shape[1] > self.n_patches:
                patches = patches[:, :self.n_patches, :]
            else:
                pad_count = self.n_patches - patches.shape[1]
                pad = torch.zeros((b, pad_count, patches.shape[-1]), device=x.device, dtype=x.dtype)
                patches = torch.cat([patches, pad], dim=1)

        # 6. Generate the Padding Mask
        real_patch_count = 1 + max(0, (original_len - self.patch_len) // self.patch_stride)
        real_patch_count = min(real_patch_count, self.n_patches)
        
        padding_mask = torch.ones((b, self.n_patches), dtype=torch.bool, device=x.device)
        padding_mask[:, :real_patch_count] = False

        return patches, padding_mask

    def forward(self, x, error_state=None):
        """
        Args:
            x: [Batch, Seq_Len, Features] - features without error history
            error_state: [Batch, error_history_len] or None; if None, zeros used
        Returns:
            Three outputs: head_15m, head_1h, head_4h
        """
        batch_size = x.shape[0]
        x_patches, padding_mask = self._build_patches(x)
        
        x_proj = self.patch_projection(x_patches)  # [Batch, n_patches, d_model]
        x_proj = self.pos_encoder(x_proj)  # Add positional encoding
        # Project error state
        if error_state is None:
            error_state = torch.zeros(batch_size, self.error_projection[0].in_features, dtype=x.dtype, device=x.device)
        error_proj = self.error_projection(error_state)  # [Batch, d_model]
        
        # Prepend error token
        error_token = error_proj.unsqueeze(1)  # [Batch, 1, d_model]
        x_with_error = torch.cat([error_token, x_proj], dim=1)  # [Batch, 1 + Seq_Len, d_model]
    

        # The error token (index 0) is never masked
        error_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([error_mask, padding_mask], dim=1) # [Batch, 1 + n_patches]

        # Pass it to the transformer
        x_enc = self.transformer(x_with_error, src_key_padding_mask=full_mask)
        
        # # Extract last step (after error token and full sequence)
        # active_mask = (~full_mask).float().unsqueeze(-1) # [Batch, Seq_Len+1, 1]

        # # Zero out the representation of padded tokens
        # masked_x_enc = x_enc * active_mask

        # # Sum the representations and divide by the count of active tokens
        # sum_representation = masked_x_enc.sum(dim=1)
        # token_counts = active_mask.sum(dim=1) # [Batch, 1]
        # global_representation = sum_representation / token_counts
        global_representation = x_enc[:, 0, :]  # Use the error token's representation as the global summary

        shared = self.shared_fc(global_representation)

        # 15m Output
        out_15m = self.head_15m(shared)
        mu_15m = out_15m[:, 0]
        sigma_15m = F.softplus(out_15m[:, 1]) + 1e-6

        # 1H Output: Conditioned on 15m
        # We pass (mu, sigma) through a learned layer to get a scale/bias vector
        cond_features_1h = self.cond_1h(out_15m) 
        shared_1h = shared + shared + torch.tanh(cond_features_1h) # Residual-style conditioning

        out_1h = self.head_1h(shared_1h)
        mu_1h = out_1h[:, 0]
        sigma_1h = F.softplus(out_1h[:, 1]) + 1e-6

        # 4H Output: Conditioned on 1H
        # Using 1H output specifically as it already contains 15m info
        cond_features_4h = self.cond_4h(out_1h)
        shared_4h = shared + shared + torch.tanh(cond_features_4h) # Residual-style conditioning

        out_4h = self.head_4h(shared_4h)
        mu_4h = out_4h[:, 0]
        sigma_4h = F.softplus(out_4h[:, 1]) + 1e-6

        # Return tuples of (prediction, volatility_confidence)
        return (mu_15m, sigma_15m), (mu_1h, sigma_1h), (mu_4h, sigma_4h)


class PBTAgent:
    def __init__(
        self,
        input_size,
        seq_len,
        error_history_len,
        lr,
        dropout,
        weight_decay,
        d_model,
        nhead,
        num_layers,
        device,
    ):
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

        base = USDCNHTransformer(
            input_size,
            seq_len,
            error_history_len,
            dropout=self.dropout,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.ema_model = copy.deepcopy(base).to(device)
        self.active_model = base.to(device)

        self.optimizer = optim.AdamW(
            self.active_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.fitness = 0.0

    def update_ema(self, decay=0.95):
        with torch.no_grad():
            for ema_param, active_param in zip(
                self.ema_model.parameters(), self.active_model.parameters()
            ):
                ema_param.data.mul_(decay).add_(active_param.data, alpha=1.0 - decay)


def layer_crossover(dst_model, parent_a, parent_b, num_layers):
    layer_groups = ["patch_projection", "error_projection", "pos_encoder", "shared_fc"]
    for i in range(num_layers):
        layer_groups.append(f"transformer.layers.{i}")
    layer_groups += [
        "abs_head",
        "direction_head",
        "volatility_head",
        "trend_head",
        "confidence_head",
        "fc1",
        "abs_output",
        "direction_output",
        "volatility_output",
        "trend_output",
        "confidence_output",
    ]

    use_a = {g: (random.random() < 0.5) for g in layer_groups}

    a_params = dict(parent_a.named_parameters())
    b_params = dict(parent_b.named_parameters())
    a_buffers = dict(parent_a.named_buffers())
    b_buffers = dict(parent_b.named_buffers())

    with torch.no_grad():
        for name, param in dst_model.named_parameters():
            src = a_params
            for group in layer_groups:
                if name == group or name.startswith(group + "."):
                    src = a_params if use_a[group] else b_params
                    break
            if name in src:
                param.data.copy_(src[name].data)

        for name, buf in dst_model.named_buffers():
            src = a_buffers
            for group in layer_groups:
                if name == group or name.startswith(group + "."):
                    src = a_buffers if use_a[group] else b_buffers
                    break
            if name in src:
                buf.data.copy_(src[name].data)
