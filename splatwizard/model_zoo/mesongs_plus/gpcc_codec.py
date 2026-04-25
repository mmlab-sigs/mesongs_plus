"""
GPCC (G-PCC / TMC3) codec for octree geometry encoding in MesonGS+.

Replaces np.savez_compressed for Morton codes (oct.npz) with GPCC lossless
octree encoding, which provides significantly better compression for 3D
point cloud geometry.

The key insight: Morton codes are interleaved 3D coordinates. GPCC's octree
encoder is specifically designed to compress 3D integer coordinates efficiently
using context-adaptive arithmetic coding with spatial priors.

Typical compression improvement: 8.6MB -> ~3-5MB for 1.5M points.
"""

import os
import subprocess
import tempfile
import pathlib
import numpy as np
from io import BytesIO
from loguru import logger

try:
    from plyfile import PlyElement, PlyData
except ImportError:
    PlyData = None
    PlyElement = None


# Default TMC3 path (relative to project root)
_DEFAULT_TMC3_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'mpeg-pcc-tmc13', 'build', 'tmc3', 'tmc3'
)
_DEFAULT_CFG_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'cfgs'
)


def _find_tmc3():
    """Find the TMC3 executable."""
    # Check environment variable first
    tmc3_env = os.environ.get('TMC3_PATH')
    # Support explicit disable: TMC3_PATH=disabled
    if tmc3_env and tmc3_env.lower() == 'disabled':
        return None
    if tmc3_env and os.path.isfile(tmc3_env):
        return tmc3_env
    
    # Check default build path
    default = os.path.abspath(_DEFAULT_TMC3_PATH)
    if os.path.isfile(default):
        return default
    
    # Check system PATH
    import shutil
    tmc3_sys = shutil.which('tmc3')
    if tmc3_sys:
        return tmc3_sys
    
    return None


def morton_to_voxel_xyz(oct, depth):
    """
    Convert Morton codes to 3D voxel coordinates.
    
    Args:
        oct: np.ndarray of Morton codes (int64)
        depth: octree depth
    
    Returns:
        xyz: np.ndarray [N, 3] of integer coordinates
    """
    occodex = (oct // (2**(depth*2))).astype(np.int32)
    occodey = ((oct - occodex.astype(np.int64) * (2**(depth*2))) // (2**depth)).astype(np.int32)
    occodez = (oct - occodex.astype(np.int64) * (2**(depth*2)) - occodey.astype(np.int64) * (2**depth)).astype(np.int32)
    return np.stack([occodex, occodey, occodez], axis=1)


def voxel_xyz_to_morton(xyz, depth):
    """
    Convert 3D voxel coordinates back to Morton codes.
    
    Args:
        xyz: np.ndarray [N, 3] of integer coordinates
        depth: octree depth
    
    Returns:
        morton: np.ndarray of Morton codes (int64)
    """
    x = xyz[:, 0].astype(np.int64)
    y = xyz[:, 1].astype(np.int64)
    z = xyz[:, 2].astype(np.int64)
    return x * (2**(depth*2)) + y * (2**depth) + z


def gpcc_encode_octree(oct, oct_param, depth, tmc3_path=None):
    """
    Encode octree data using GPCC lossless geometry encoder.
    
    This replaces:
        np.savez_compressed(path, points=oct, params=oct_param)
    
    Args:
        oct: np.ndarray, Morton codes (1D int array, sorted)
        oct_param: np.ndarray, boundary parameters [minx,maxx,miny,maxy,minz,maxz]
        depth: int, octree depth
        tmc3_path: str or None, path to tmc3 executable
    
    Returns:
        bitstream: bytes, the GPCC encoded bitstream
        
    The bitstream format:
        [4 bytes] depth (int32)
        [48 bytes] oct_param (6 x float64)
        [4 bytes] N (int32, number of points)
        [4 bytes] gpcc_len (int32, length of GPCC bitstream)
        [gpcc_len bytes] GPCC compressed geometry
    """
    if tmc3_path is None:
        tmc3_path = _find_tmc3()
    if tmc3_path is None:
        raise FileNotFoundError(
            "TMC3 executable not found. Please set TMC3_PATH environment variable "
            "or build TMC3 from https://github.com/MPEGGroup/mpeg-pcc-tmc13"
        )
    
    tmc3_path = os.path.abspath(tmc3_path)
    cfg_dir = os.path.abspath(_DEFAULT_CFG_DIR)
    encoder_cfg = os.path.join(cfg_dir, 'lossless_encoder.cfg')
    
    N = len(oct)
    
    # Convert Morton codes to voxel xyz
    xyz = morton_to_voxel_xyz(oct, depth)
    
    # Create PLY file for TMC3 input
    dtype_full = [(attr, 'f4') for attr in ('x', 'y', 'z')]
    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, xyz.astype(np.float32)))
    el = PlyElement.describe(elements, 'vertex')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, 'oct_pc.ply')
        bin_path = os.path.join(tmpdir, 'oct_compressed.drc')
        PlyData([el]).write(ply_path)
        
        # Run TMC3 encoder
        result = subprocess.run(
            [
                tmc3_path,
                '-c', encoder_cfg,
                f'--uncompressedDataPath={ply_path}',
                f'--compressedStreamPath={bin_path}'
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"TMC3 encoder failed: {result.stderr}")
            raise RuntimeError(f"TMC3 encoder failed with code {result.returncode}")
        
        with open(bin_path, 'rb') as f:
            gpcc_bitstream = f.read()
    
    gpcc_size = len(gpcc_bitstream)
    logger.info(f"GPCC encoded octree: {N} points -> {gpcc_size} bytes ({gpcc_size/1024/1024:.2f} MB)")
    
    # Pack into our format: header + GPCC bitstream
    buf = BytesIO()
    buf.write(np.array([depth], dtype=np.int32).tobytes())          # 4 bytes
    buf.write(oct_param.astype(np.float64).tobytes())                # 48 bytes
    buf.write(np.array([N], dtype=np.int32).tobytes())               # 4 bytes
    buf.write(np.array([gpcc_size], dtype=np.int32).tobytes())       # 4 bytes
    buf.write(gpcc_bitstream)                                        # gpcc_size bytes
    
    buf.seek(0)
    return buf.read()


def gpcc_decode_octree(bitstream, tmc3_path=None):
    """
    Decode octree data from GPCC bitstream.
    
    This replaces:
        data = np.load('oct.npz')
        octree = data['points']
        oct_param = data['params']
    
    Args:
        bitstream: bytes, as produced by gpcc_encode_octree
        tmc3_path: str or None, path to tmc3 executable
    
    Returns:
        oct: np.ndarray, Morton codes
        oct_param: np.ndarray, boundary parameters
        depth: int
    """
    if tmc3_path is None:
        tmc3_path = _find_tmc3()
    if tmc3_path is None:
        raise FileNotFoundError("TMC3 executable not found")
    
    tmc3_path = os.path.abspath(tmc3_path)
    cfg_dir = os.path.abspath(_DEFAULT_CFG_DIR)
    decoder_cfg = os.path.join(cfg_dir, 'decoder.cfg')
    
    buf = BytesIO(bitstream)
    
    # Read header
    depth = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    oct_param = np.frombuffer(buf.read(48), dtype=np.float64).copy()
    N = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    gpcc_size = int(np.frombuffer(buf.read(4), dtype=np.int32)[0])
    gpcc_bitstream = buf.read(gpcc_size)
    
    # Decode with TMC3
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, 'oct_compressed.drc')
        ply_path = os.path.join(tmpdir, 'oct_pc_decoded.ply')
        
        with open(bin_path, 'wb') as f:
            f.write(gpcc_bitstream)
        
        result = subprocess.run(
            [
                tmc3_path,
                '-c', decoder_cfg,
                f'--compressedStreamPath={bin_path}',
                f'--reconstructedDataPath={ply_path}'
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"TMC3 decoder failed: {result.stderr}")
            raise RuntimeError(f"TMC3 decoder failed with code {result.returncode}")
        
        plydata = PlyData.read(ply_path)
        xyz = np.stack([
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ], axis=1).astype(np.int32)
    
    # Convert back to Morton codes
    morton = voxel_xyz_to_morton(xyz, depth)
    
    # Sort by Morton code (GPCC may change ordering)
    sort_idx = np.argsort(morton)
    morton = morton[sort_idx]
    
    assert len(morton) == N, f"Point count mismatch: decoded {len(morton)}, expected {N}"
    
    logger.info(f"GPCC decoded octree: {N} points, depth={depth}")
    
    return morton, oct_param, depth


def gpcc_get_encoded_size(oct, oct_param, depth, tmc3_path=None):
    """
    Get the GPCC encoded size without saving to final location.
    Useful for size estimation in qbit_search_tool.
    
    Returns:
        size_bytes: int
    """
    bitstream = gpcc_encode_octree(oct, oct_param, depth, tmc3_path)
    return len(bitstream)


def is_gpcc_available():
    """Check if GPCC (TMC3) is available."""
    return _find_tmc3() is not None
