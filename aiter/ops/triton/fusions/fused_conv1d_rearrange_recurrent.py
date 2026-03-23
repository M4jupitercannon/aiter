# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_conv1d_rearrange_recurrent import _fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule_kernel

PAD_SLOT_ID = -1

def fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    g: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused kernel that combines:
    1. causal_conv1d_update: 1D causal convolution on qkv
    2. rearrange + gated delta rule: recurrent attention computation
    
    This fusion reduces memory traffic by avoiding intermediate tensor materialization.
    Uses a single Triton kernel for decode-only path.
    
    Args:
        qkv (torch.Tensor):
            mixed qkv of shape: `[L, H_qk*D_qk + H_qk*D_qk + H_v*D_v]`.
            queries of shape `[L, H_qk*D_qk]`.
            keys of shape    `[L, H_qk*D_qk]`.
            values of shape  `[L, H_v*D_v]`.
        conv_state: (..., dim, state_len), where state_len >= width - 1, for conv1d
        weight: (dim, width), for conv1d
        bias: (dim,), for conv1d
        conv_state_indices: (batch,), dtype int32, for conv1d
            If not None, the conv_state is a larger tensor along the batch dim,
            and we are selecting the batch coords specified by conv_state_indices.
            Useful for a continuous batching scenario.
        initial_state_idx: (batch,), dtype int32, for conv1d
            The pointer into conv_state_indices, where the cache block containing the initial state is located.
        pad_slot_id: int, for conv1d
                if conv_state_indices is passed, lets the kernel identify padded
                entries that will not be processed,
                for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
                in this case, the kernel will not process entries at
                indices 0 and 3
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        key_dim, value_dim: Dimensions for key and value
        head_k_dim, head_v_dim: Per-head dimensions
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q, k
        
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, V, K]`.
    """
    ## for conv1d
    if validate_data:
        assert pad_slot_id is not None
        assert qkv.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    batch, dim = qkv.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this

    stride_w_dim, stride_w_width = weight.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    assert num_accepted_tokens is None, "num_accepted_tokens should be None in the fused kernel"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    ## for gated delta rule
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    assert qkv.shape == expected_shape, f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"
    if scale is None:
        scale = head_k_dim ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        h_q = key_dim // head_k_dim
        beta = torch.ones([1, qkv.shape[0], h_q], dtype=qkv.dtype, device=qkv.device)

    # decode-only mode (single token per sequence)
    assert (
        cu_seqlens is not None 
        and batch == len(cu_seqlens) - 1
    ), f"cu_seqlens {str(cu_seqlens)} is not compatible for decode mode"
    
    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = qkv.new_empty(NK, B, T, HV, V)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = qkv.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    grid = (NK, NV, N * HV)
    _fused_causal_conv1d_update_rearrange_recurrent_gated_delta_rule_kernel[grid](
        qkv=qkv,
        ## for conv1d
        weight=weight,
        bias=bias,
        conv_state=conv_state,
        conv_state_indices=conv_state_indices,
        state_len=state_len,
        num_cache_lines=num_cache_lines,
        stride_w_dim=stride_w_dim,
        stride_w_width=stride_w_width,
        stride_conv_state_seq=stride_istate_seq,
        stride_conv_state_dim=stride_istate_dim,
        stride_conv_state_tok=stride_istate_token,
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        ## for gated delta rule
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == 4, # v.ndim
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        # for conv1d META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        # others
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state

