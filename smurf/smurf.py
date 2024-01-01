from typing import Optional, Tuple

import torch
import torch.nn as nn

from ._raft import _raft, ResidualBlock

__all__: Tuple[str] = ("raft_smurf",)


def raft_smurf(checkpoint: Optional[str] = None) -> nn.Module:
    """Builds the RAFT (large) SMURF model.

    Args:
        checkpoint (Optional[str]): PyTorch checkpoint to be loaded.

    Returns:
        model (nn.Module): RAFT (large) SMURF model as a PyTorch Module.
    """
    # Init model
    model: nn.Module = _raft(
        weights=None,
        progress=False,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_block=ResidualBlock,
        feature_encoder_norm_layer=nn.InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_block=ResidualBlock,
        context_encoder_norm_layer=nn.InstanceNorm2d,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(256, 192),
        motion_encoder_flow_layers=(128, 64),
        motion_encoder_out_channels=128,
        # Recurrent block
        recurrent_block_hidden_state_size=128,
        recurrent_block_kernel_size=((1, 5), (5, 1)),
        recurrent_block_padding=((0, 2), (2, 0)),
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        use_mask_predictor=True,
    )
    # Load checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model
