"""
Laplace Range Coder for SegQuant quantized indices.

Uses constriction library's QuantizedLaplace model to encode/decode
quantized indices with per-block Laplace distribution parameters.

Key idea: RAHT AC coefficients and codebook quantized indices follow
approximately Laplace distributions. By estimating per-block (loc, scale)
parameters and using range coding, we achieve better compression than
generic zlib (np.savez_compressed).
"""

import numpy as np
import constriction
import torch
from io import BytesIO


def estimate_laplace_params_per_block(q_indices_flat, split, n_channels):
    """
    Estimate per-block Laplace distribution parameters for quantized indices.
    
    Args:
        q_indices_flat: np.ndarray, flattened quantized indices [N * C] or [C, N]
                        If 1D, will be reshaped to [C, N] using n_channels.
        split: list of int, block lengths (sum = N)
        n_channels: int, number of channels
    
    Returns:
        locs: np.ndarray [C, n_block], per-block location (median)
        scales: np.ndarray [C, n_block], per-block scale (MAD-based)
    """
    N = sum(split)
    n_block = len(split)
    
    if q_indices_flat.ndim == 1:
        q_data = q_indices_flat.reshape(n_channels, N)
    else:
        q_data = q_indices_flat  # already [C, N]
    
    locs = np.zeros((n_channels, n_block), dtype=np.float64)
    scales_out = np.zeros((n_channels, n_block), dtype=np.float64)
    
    start = 0
    for b in range(n_block):
        length = split[b]
        block = q_data[:, start:start+length].astype(np.float64)  # [C, length]
        
        for c in range(n_channels):
            med = np.median(block[c])
            locs[c, b] = med
            # MAD (Median Absolute Deviation) -> Laplace scale
            mad = np.median(np.abs(block[c] - med))
            # Ensure scale > 0 (constriction requires positive scale)
            scales_out[c, b] = max(mad, 0.5)
        
        start += length
    
    return locs, scales_out


def laplace_encode_blocks(q_indices_flat, split, n_channels, max_val=None, auto_model=False):
    """
    Encode quantized indices using per-block Laplace range coding.
    
    Args:
        q_indices_flat: np.ndarray, flattened quantized indices.
                        For [N, C] data from pure_quant_wo_minmax, 
                        pass q_indices.cpu().numpy() which is [C, N] after transpose,
                        or flatten and specify n_channels.
        split: list of int, block lengths
        n_channels: int
        max_val: int or None, maximum symbol value. If None, inferred from data.
        auto_model: bool, if True, try both Laplace and Gaussian per-block and pick the smaller one.
    
    Returns:
        bitstream: bytes, the encoded bitstream
        header: dict with metadata needed for decoding
    """
    N = sum(split)
    n_block = len(split)
    
    if q_indices_flat.ndim == 1:
        q_data = q_indices_flat.reshape(n_channels, N)
    else:
        q_data = q_indices_flat.copy()
    
    if max_val is None:
        max_val = int(q_data.max())
    min_val = int(q_data.min())
    
    # Estimate Laplace parameters
    locs, scales = estimate_laplace_params_per_block(q_data, split, n_channels)
    
    # Create Laplace model
    model_family = constriction.stream.model.QuantizedLaplace(min_val, max_val + 1)
    
    # Build per-symbol loc and scale arrays for Laplace
    all_symbols = []
    all_locs = []
    all_scales = []
    
    for c in range(n_channels):
        start = 0
        for b in range(n_block):
            length = split[b]
            block_syms = q_data[c, start:start+length]
            all_symbols.append(block_syms)
            all_locs.append(np.full(length, locs[c, b], dtype=np.float64))
            all_scales.append(np.full(length, scales[c, b], dtype=np.float64))
            start += length
    
    all_symbols = np.concatenate(all_symbols).astype(np.int32)
    all_locs = np.concatenate(all_locs)
    all_scales = np.concatenate(all_scales)
    
    if auto_model:
        # Try Laplace encoding
        laplace_encoder = constriction.stream.queue.RangeEncoder()
        laplace_encoder.encode(all_symbols, model_family, all_locs, all_scales)
        laplace_compressed = laplace_encoder.get_compressed()
        laplace_size = len(laplace_compressed) * 4  # uint32 words → bytes
        
        # Try Gaussian encoding
        try:
            gauss_locs, gauss_scales = _estimate_gaussian_params_per_block(q_data, split, n_channels)
            g_locs_arr, g_scales_arr = _build_gaussian_param_arrays(q_data, split, n_channels)
            gauss_model_family = constriction.stream.model.QuantizedGaussian(min_val, max_val + 1)
            gauss_encoder = constriction.stream.queue.RangeEncoder()
            gauss_encoder.encode(all_symbols, gauss_model_family, g_locs_arr, g_scales_arr)
            gauss_compressed = gauss_encoder.get_compressed()
            gauss_size = len(gauss_compressed) * 4
        except Exception as e:
            print(f"  Auto entropy: Gaussian encoding failed ({e}), using Laplace")
            gauss_size = float('inf')
        
        if gauss_size < laplace_size:
            print(f"  Auto entropy: Gaussian wins ({gauss_size} vs {laplace_size} bytes, saving {laplace_size-gauss_size} bytes)")
            compressed = gauss_compressed
            model_flag = 1  # Gaussian
            # Use Gaussian params for header
            locs = gauss_locs
            scales = gauss_scales
        else:
            saving = gauss_size - laplace_size if gauss_size != float('inf') else 0
            print(f"  Auto entropy: Laplace wins ({laplace_size} vs {gauss_size} bytes)")
            compressed = laplace_compressed
            model_flag = 0  # Laplace
    else:
        # Encode with Laplace only
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(all_symbols, model_family, all_locs, all_scales)
        compressed = encoder.get_compressed()  # np.ndarray of uint32
        model_flag = 0
    
    # Pack into bytes
    buf = BytesIO()
    # Header: min_val(4), max_val(4), n_channels(4), n_block(4), N(4)
    buf.write(np.array([min_val], dtype=np.int32).tobytes())
    buf.write(np.array([max_val], dtype=np.int32).tobytes())
    buf.write(np.array([n_channels], dtype=np.int32).tobytes())
    buf.write(np.array([n_block], dtype=np.int32).tobytes())
    buf.write(np.array([N], dtype=np.int32).tobytes())
    # Model flag (for auto_model decoding)
    buf.write(np.array([model_flag], dtype=np.int32).tobytes())
    # Split lengths
    buf.write(np.array(split, dtype=np.int32).tobytes())
    # Laplace params: locs and scales [C, n_block] each as float64
    buf.write(locs.tobytes())
    buf.write(scales.tobytes())
    # Compressed data length (in uint32 words)
    buf.write(np.array([len(compressed)], dtype=np.int32).tobytes())
    # Compressed data
    buf.write(compressed.tobytes())
    
    buf.seek(0)
    return buf.read()


def _estimate_gaussian_params_per_block(q_data, split, n_channels):
    """Estimate per-block Gaussian (mean, std) parameters."""
    N = sum(split)
    n_block = len(split)
    
    locs = np.zeros((n_channels, n_block), dtype=np.float64)
    scales_out = np.zeros((n_channels, n_block), dtype=np.float64)
    
    start = 0
    for b in range(n_block):
        length = split[b]
        block = q_data[:, start:start+length].astype(np.float64)
        
        for c in range(n_channels):
            locs[c, b] = np.mean(block[c])
            std = np.std(block[c])
            scales_out[c, b] = max(std, 0.5)
        
        start += length
    
    return locs, scales_out


def _select_best_model_per_block(q_data, split, n_channels, min_val, max_val,
                                  lap_locs, lap_scales, gauss_locs, gauss_scales):
    """Select best model (Laplace=0, Gaussian=1) per block based on log-likelihood."""
    n_block = len(split)
    choices = np.zeros((n_channels, n_block), dtype=np.int32)
    
    start = 0
    for b in range(n_block):
        length = split[b]
        block = q_data[:, start:start+length].astype(np.float64)
        
        for c in range(n_channels):
            data = block[c]
            # Laplace log-likelihood
            lap_ll = -np.sum(np.abs(data - lap_locs[c, b]) / max(lap_scales[c, b], 0.5))
            # Gaussian log-likelihood  
            gauss_ll = -np.sum((data - gauss_locs[c, b])**2 / (2 * max(gauss_scales[c, b], 0.5)**2))
            
            if gauss_ll > lap_ll:
                choices[c, b] = 1  # Gaussian is better
        
        start += length
    
    return choices


def _build_gaussian_param_arrays(q_data, split, n_channels):
    """Build per-symbol Gaussian parameter arrays for encoding."""
    N = sum(split)
    n_block = len(split)
    
    gauss_locs, gauss_scales = _estimate_gaussian_params_per_block(q_data, split, n_channels)
    
    all_locs = []
    all_scales = []
    
    for c in range(n_channels):
        start = 0
        for b in range(n_block):
            length = split[b]
            all_locs.append(np.full(length, gauss_locs[c, b], dtype=np.float64))
            all_scales.append(np.full(length, gauss_scales[c, b], dtype=np.float64))
            start += length
    
    return np.concatenate(all_locs), np.concatenate(all_scales)


def _auto_dtype(min_val, max_val):
    """根据 min/max 值范围自动选择最小的 numpy dtype。"""
    if min_val >= 0:
        if max_val <= 255:
            return np.uint8
        elif max_val <= 65535:
            return np.uint16
        else:
            return np.int32
    else:
        if min_val >= -128 and max_val <= 127:
            return np.int8
        elif min_val >= -32768 and max_val <= 32767:
            return np.int16
        else:
            return np.int32


def laplace_decode_blocks(bitstream, dtype=None):
    """
    Decode quantized indices from a Laplace range-coded bitstream.
    
    Args:
        bitstream: bytes, as produced by laplace_encode_blocks
        dtype: target dtype for output indices. If None, auto-detect from
               min_val/max_val in the header (recommended).
    
    Returns:
        q_data: np.ndarray [C, N], decoded quantized indices
        split: list of int, block lengths
        n_channels: int
    """
    buf = BytesIO(bitstream)
    
    # Read header
    min_val = np.frombuffer(buf.read(4), dtype=np.int32)[0]
    max_val = np.frombuffer(buf.read(4), dtype=np.int32)[0]
    n_channels = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    n_block = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    N = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    
    # Model flag (0=Laplace, 1=Gaussian, added in improvement experiments)
    model_flag = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    
    # Split lengths
    split = np.frombuffer(buf.read(n_block * 4), dtype=np.int32).tolist()
    
    # Laplace params
    locs = np.frombuffer(buf.read(n_channels * n_block * 8), dtype=np.float64).reshape(n_channels, n_block)
    scales = np.frombuffer(buf.read(n_channels * n_block * 8), dtype=np.float64).reshape(n_channels, n_block)
    
    # Compressed data
    compressed_len = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    compressed = np.frombuffer(buf.read(compressed_len * 4), dtype=np.uint32).copy()
    
    # Build per-symbol loc and scale arrays
    if model_flag == 1:
        model_family = constriction.stream.model.QuantizedGaussian(int(min_val), int(max_val) + 1)
    else:
        model_family = constriction.stream.model.QuantizedLaplace(int(min_val), int(max_val) + 1)
    
    all_locs = []
    all_scales = []
    total_symbols = n_channels * N
    
    for c in range(n_channels):
        start = 0
        for b in range(n_block):
            length = split[b]
            all_locs.append(np.full(length, locs[c, b], dtype=np.float64))
            all_scales.append(np.full(length, scales[c, b], dtype=np.float64))
            start += length
    
    all_locs = np.concatenate(all_locs)
    all_scales = np.concatenate(all_scales)
    
    # Decode
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded = decoder.decode(model_family, all_locs, all_scales)
    
    # Auto-select dtype if not specified
    if dtype is None:
        dtype = _auto_dtype(int(min_val), int(max_val))
    
    # Reshape back to [C, N]
    q_data = decoded.reshape(n_channels, N).astype(dtype)
    
    return q_data, split, n_channels


def laplace_encode_2d(q_indices_2d, split, max_val=None):
    """
    Convenience wrapper: encode a [C, N] or [N, C] quantized index array.
    Assumes input is [C, N] (channels-first, as output by pure_quant_wo_minmax).
    
    Args:
        q_indices_2d: np.ndarray [C, N]
        split: list of block lengths (sum = N)
        max_val: optional max symbol value
    
    Returns:
        bitstream: bytes
    """
    n_channels = q_indices_2d.shape[0]
    return laplace_encode_blocks(q_indices_2d, split, n_channels, max_val)


def laplace_decode_2d(bitstream, dtype=None):
    """
    Convenience wrapper: decode back to [C, N] quantized indices.
    
    Args:
        bitstream: bytes
        dtype: target dtype (None = auto-detect from header)
    
    Returns:
        q_data: np.ndarray [C, N]
        split: list of block lengths
    """
    q_data, split, n_channels = laplace_decode_blocks(bitstream, dtype)
    return q_data, split


def encode_array_to_bytes(array, split, n_channels):
    """
    Encode a flattened quantized array to bytes using Laplace range coding.
    This is the main entry point for replacing np.savez_compressed.
    
    Args:
        array: np.ndarray, the quantized indices (1D flattened or 2D [C, N])
        split: list of block lengths
        n_channels: number of channels
    
    Returns:
        bytes: encoded bitstream
    """
    return laplace_encode_blocks(array, split, n_channels)


def decode_bytes_to_array(bitstream, dtype=None):
    """
    Decode bytes back to quantized indices array.
    
    Args:
        bitstream: bytes
        dtype: target numpy dtype (None = auto-detect from header)
    
    Returns:
        q_data: np.ndarray [C, N]
        split: list of block lengths
        n_channels: int
    """
    return laplace_decode_blocks(bitstream, dtype)


def get_encoded_size(q_indices, split, n_channels, max_val=None):
    """
    Get the compressed size in bytes without writing to disk.
    Useful for ILP size estimation in qbit_search_tool.
    
    Args:
        q_indices: quantized indices array
        split: block lengths
        n_channels: number of channels
        max_val: optional max value
    
    Returns:
        size_bytes: int, total compressed size in bytes
    """
    bitstream = laplace_encode_blocks(q_indices, split, n_channels, max_val)
    return len(bitstream)
