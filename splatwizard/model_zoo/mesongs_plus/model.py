import io
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import typing
import math

import numpy as np
import torch
from einops import repeat
from plyfile import PlyData, PlyElement
from torch import nn
from torch_scatter import scatter_max
from loguru import logger
from tqdm import tqdm
import tempfile
import zipfile
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml 

from splatwizard.compression.entropy_codec import ArithmeticCodec
from splatwizard.model_zoo.mesongs_plus.laplace_codec import (
    laplace_encode_blocks, laplace_decode_blocks, get_encoded_size
)
from splatwizard.model_zoo.mesongs_plus.ntk_codec import ntk_encode, ntk_decode
from splatwizard.model_zoo.mesongs_plus.gpcc_codec import (
    gpcc_encode_octree, gpcc_decode_octree, is_gpcc_available
)
from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.metrics.loss_utils import l1_func, ssim_func
from splatwizard.common.constants import BIT2MB_SCALE
from splatwizard.rasterizer.meson_gs import GaussianRasterizationSettings, GaussianRasterizer, GaussianRasterizerIndexed
from splatwizard.model_zoo.mesongs_plus.config import MesonGSPlusModelParams, MesonGSPlusOptimizationParams
from splatwizard.config import PipelineParams
from splatwizard.model_zoo.mesongs_plus.meson_utils import VanillaQuan, vq_features, split_length, pure_quant_wo_minmax
from splatwizard.model_zoo.mesongs_plus.qbit_search_tool import search_qbits
from splatwizard.model_zoo.mesongs_plus.raht_torch import haar3D_param, inv_haar3D_param, transform_batched_torch, itransform_batched_torch, copyAsort
from splatwizard.modules.dataclass import RenderResult, LossPack
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard._cmod.simple_knn import distCUDA2    # noqa
from splatwizard.scheduler import Scheduler, task
from splatwizard.utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric
)
from splatwizard.utils.graphics_utils import BasicPointCloud
from splatwizard.utils.sh_utils import eval_sh
from splatwizard.compression.entropy_model import EntropyGaussian
from splatwizard.compression.quantizer import STE_binary, STE_multistep, Quantize_anchor, UniformQuantizer, STEQuantizer
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.dataclass import RenderResult

from ...scene import CameraIterator

@dataclass
class MesonGSPlusRenderResult(RenderResult):
    imp: typing.Any = None
    
def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)
                       
def octreecodes(ppoints, pdepht, merge_type='mean',imps=None):
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
    otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
    otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    
    ki_ranks = np.argsort(ki)
    ppoints = ppoints[ki_ranks]
    ki = ki[ki_ranks]

    ppoints = np.concatenate([ki.reshape(-1, 1), ppoints], -1)
    # print('here 4', ppoints.shape)
    dedup_points = np.split(ppoints[:, 1:], np.unique(ki, return_index=True)[1][1:])
    
    # print('ki.shape', ki.shape)
    
    # print('ki.shape', ki.shape)
    final_feature = []
    imp_merge_count = 0  # Debug: count how many times we use max for importance
    if merge_type == 'mean':
        for dedup_point in dedup_points:
            # print(np.mean(dedup_point, 0).shape)
            merged = np.mean(dedup_point, 0).reshape(1, -1)
            # Special handling for importance (last column): use max instead of mean
            # to preserve high-importance points during octree merging
            # dedup_point shape: [n_merged_points, 60] where 60 = xyz(3) + features(57 with imp)
            if dedup_point.shape[1] >= 60:  # Has importance as the last dimension
                if dedup_point.shape[0] > 1:  # Only matters when merging multiple points
                    imp_merge_count += 1
                merged[0, -1] = np.max(dedup_point[:, -1])  # Use max for importance
            final_feature.append(merged)
        if imp_merge_count > 0:
            print(f"Octree merged {imp_merge_count} groups using max importance")
    elif merge_type == 'imp':
        dedup_imps = np.split(imps, np.unique(ki, return_index=True)[1][1:])
        for dedup_point, dedup_imp in zip(dedup_points, dedup_imps):
            dedup_imp = dedup_imp.reshape(1, -1)
            if dedup_imp.shape[-1] == 1:
                # print('dedup_point.shape', dedup_point.shape)
                final_feature.append(dedup_point)
            else:
                # print('dedup_point.shape, dedup_imp.shape', dedup_point.shape, dedup_imp.shape)
                fdp = (dedup_imp / np.sum(dedup_imp)) @ dedup_point
                # print('fdp.shape', fdp.shape)
                final_feature.append(fdp)
    elif merge_type == 'rand':
        for dedup_point in dedup_points:
            ld = len(dedup_point)
            id = torch.randint(0, ld, (1,))[0]
            final_feature.append(dedup_point[id].reshape(1, -1))
    else:
        raise NotImplementedError
    ki = np.unique(ki)
    final_feature = np.concatenate(final_feature, 0)
    # print('final_feature.shape', final_feature.shape)
    return (ki,minx,maxx,miny,maxy,minz,maxz, final_feature)


def create_octree_overall(ppoints, pfeatures, imp, depth, oct_merge):
    ori_points_num = ppoints.shape[0]
    ppoints = np.concatenate([ppoints, pfeatures], -1)
    occ=octreecodes(ppoints, depth, oct_merge, imp)
    final_points_num = occ[0].shape[0]
    occodex=(occ[0]/(2**(depth*2))).astype(int)
    occodey=((occ[0]-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(occ[0]-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)
    voxel_xyz = np.array([occodex,occodey,occodez], dtype=int).T
    features = occ[-1][:, 3:]
    paramarr=np.asarray([occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) # boundary
    # print('oct[0]', type(oct[0]))
    return voxel_xyz, features, occ[0], paramarr, ori_points_num, final_points_num

def decode_oct(paramarr, oct, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    occodex=(oct/(2**(depth*2))).astype(int)
    occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    V = np.array([occodex,occodey,occodez], dtype=int).T
    koorx=xletra[occodex]
    koory=yletra[occodey]
    koorz=zletra[occodez]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V

def ToEulerAngles_FT(q):

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = torch.sqrt(1 + 2 * (w * y - x * z))
    cosp = torch.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * torch.arctan2(sinp, cosp) - torch.pi / 2
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.arctan2(siny_cosp, cosy_cosp)

    roll = roll.reshape(-1, 1)
    pitch = pitch.reshape(-1, 1)
    yaw = yaw.reshape(-1, 1)

    return torch.concat([roll, pitch, yaw], -1)

def seg_quant_ave(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    for length in split:
        outs.append(qas[cnt](x[start:start+length]))
        cnt += 1
        start += length
    return torch.concat(outs, dim=0)

def get_dtype_for_bits(num_bits):
    if num_bits <= 8:
        return np.uint8
    elif num_bits <= 16:
        return np.uint16
    elif num_bits <= 32:
        return np.uint32
    else:
        return np.uint64

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x

def torch_vanilla_quant_ave(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    trans = []
    for length in split:
        i_scale = qas[cnt].scale
        i_zp = qas[cnt].zero_point
        i_bit = qas[cnt].bit
        outs.append(quantize_tensor(
            x[start:start+length], 
            scale=i_scale,
            zero_point=i_zp,
            num_bits=i_bit).cpu().numpy()) 
        trans.extend([i_scale.item(), i_zp.item()])
        cnt += 1
        start += length
    return np.concatenate(outs, axis=0), trans

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

def torch_vanilla_dequant_ave(x, split, sz):
    cnt = 0 
    start = 0
    outs = []
    for length in split:
        i_scale = sz[cnt]
        i_zp = sz[cnt+1]
        outs.append(
            dequantize_tensor(
                x[start:start+length],
                scale=i_scale,
                zero_point=i_zp
            )
        )
        cnt+=2
        start += length
    return torch.concat(outs, axis=0)


class MesonGSPlus(LossMixin, DensificationAndPruneMixin, GaussianModel):

    def setup_functions(self):
        def build_rotation_from_euler(roll, pitch, yaw):
            R = torch.zeros((roll.size(0), 3, 3), device='cuda')

            R[:, 0, 0] = torch.cos(pitch) * torch.cos(roll)
            R[:, 0, 1] = -torch.cos(yaw) * torch.sin(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
            R[:, 0, 2] = torch.sin(yaw) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
            R[:, 1, 0] = torch.cos(pitch) * torch.sin(roll)
            R[:, 1, 1] = torch.cos(yaw) * torch.cos(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
            R[:, 1, 2] = -torch.sin(yaw) * torch.cos(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
            R[:, 2, 0] = -torch.sin(pitch)
            R[:, 2, 1] = torch.sin(yaw) * torch.cos(pitch)
            R[:, 2, 2] = torch.cos(yaw) * torch.cos(pitch)

            return R


        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def safe_euler(euler):
            # 把 NaN 转为 0，但保持 requires_grad
            euler = torch.where(torch.isnan(euler), torch.zeros_like(euler), euler)

            # 限制角度范围，防止 sin/cos 溢出造成 inf
            euler = torch.clamp(euler, min=-1e6, max=1e6)
            return euler

        def build_covariance_from_scaling_euler(scaling, scaling_modifier, euler, return_symm=True):
            euler = safe_euler(euler)
            s = scaling_modifier * scaling
            L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
            R = build_rotation_from_euler(euler[:, 2], euler[:, 1], euler[:, 0])

            L[:,0,0] = s[:,0]
            L[:,1,1] = s[:,1]
            L[:,2,2] = s[:,2]

            L = R @ L
            actual_covariance = L @ L.transpose(1, 2)
            if return_symm:
                symm = strip_symmetric(actual_covariance)
                return symm
            else:
                return actual_covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_activation_for_euler = build_covariance_from_scaling_euler
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, model_param: MesonGSPlusModelParams):
        logger.info('mesongs start init here')
        super().__init__()
        self.finetune_lr_scale = model_param.finetune_lr_scale
        self.num_bits = model_param.num_bits
        self.depth = model_param.depth
        self.percent = model_param.percent
        self.raht = model_param.raht
        self.merge_type = model_param.oct_merge
        self.debug = model_param.debug
        self.clamp_color = model_param.clamp_color
        self.per_channel_quant = model_param.per_channel_quant
        self.per_block_quant = model_param.per_block_quant
        self.use_indexed = model_param.use_indexed
        self.scene_imp = model_param.scene_imp
        self.n_block = model_param.n_block
        self.codebook_size = model_param.codebook_size
        self.batch_size = model_param.batch_size
        self.steps = model_param.steps
        self.use_quat = model_param.use_quat
        self.size_limit_mb = model_param.size_limit_mb
        self.sh_keep_threshold = model_param.sh_keep_threshold
        self.sh_keep_topk = model_param.sh_keep_topk
        self.enable_golden_search = model_param.enable_golden_search
        self.golden_search_interval = model_param.golden_search_interval
        self.enable_binary_search = model_param.enable_binary_search
        self.binary_search_interval = model_param.binary_search_interval
        self.obj_threshold_ratio = model_param.obj_threshold_ratio
        self.cb_quant_bits = model_param.cb_quant_bits
        if model_param.yaml_path != "":
            with open(model_param.yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
        self.depth = config_dict["depth"]
        
        # 命令行参数优先于yaml配置（检查是否为非默认值）
        # percent: 默认值为0.66
        if abs(model_param.percent - 0.66) > 0.001:
            self.percent = model_param.percent
            logger.info(f"using CLI percent: {self.percent}")
        else:
            self.percent = config_dict["prune"]
            logger.info(f"using yaml prune: {self.percent}")
        
        if model_param.codebook_size != 2048:
            self.codebook_size = model_param.codebook_size
            logger.info(f"using CLI codebook_size: {self.codebook_size}")
        else:
            self.codebook_size = config_dict["cb"]
            logger.info(f"using yaml codebook_size: {self.codebook_size}")
        
        # n_block: 默认值为66
        if model_param.n_block != 66:
            self.n_block = model_param.n_block
            logger.info(f"using CLI n_block: {self.n_block}")
        else:
            self.n_block = config_dict["n_block"]
            logger.info(f"using yaml n_block: {self.n_block}")
        
        # num_bits: 默认值为8，yaml中可能没有此配置
        if model_param.num_bits != 8:
            self.num_bits = model_param.num_bits
            logger.info(f"using CLI num_bits: {self.num_bits}")
        elif "num_bits" in config_dict:
            self.num_bits = config_dict["num_bits"]
            logger.info(f"using yaml num_bits: {self.num_bits}")
        else:
            logger.info(f"using default num_bits: {self.num_bits}")
        
        self.finetune_lr_scale = config_dict["finetune_lr_scale"]
        if "size_limit_mb" in config_dict:
            self.size_limit_mb = config_dict["size_limit_mb"]
        
        # Eval-time re-pruning rate
        self.pruning_rate = model_param.pruning_rate
        if "pruning_rate" in config_dict:
            self.pruning_rate = config_dict["pruning_rate"]
        # CLI override (non-default value takes priority)
        if abs(model_param.pruning_rate - (-1.0)) > 0.001:
            self.pruning_rate = model_param.pruning_rate
            logger.info(f"using CLI pruning_rate: {self.pruning_rate}")
        
        # cb_quant_bits: codebook/kept points 量化位宽
        if model_param.cb_quant_bits != 8:
            self.cb_quant_bits = model_param.cb_quant_bits
            logger.info(f"using CLI cb_quant_bits: {self.cb_quant_bits}")
        elif "cb_quant_bits" in config_dict:
            self.cb_quant_bits = config_dict["cb_quant_bits"]
            logger.info(f"using yaml cb_quant_bits: {self.cb_quant_bits}")
        else:
            logger.info(f"using default cb_quant_bits: {self.cb_quant_bits}")
        
        # ntk_n_block / cb_n_block: independent block counts for NTK and codebook Laplace encoding
        self.ntk_n_block = model_param.ntk_n_block if model_param.ntk_n_block > 0 else self.n_block
        self.cb_n_block = model_param.cb_n_block if model_param.cb_n_block > 0 else self.n_block
        if model_param.ntk_n_block > 0:
            logger.info(f"using independent ntk_n_block: {self.ntk_n_block}")
        if model_param.cb_n_block > 0:
            logger.info(f"using independent cb_n_block: {self.cb_n_block}")

        # num_keep: eval-time kept point count (-1 = use default logic)
        self.num_keep = model_param.num_keep
        if self.num_keep >= 0:
            logger.info(f"using CLI num_keep: {self.num_keep}")
        else:
            logger.info(f"num_keep not set, will use default adjustment logic")
        
        # Improvement experiment flags
        self.channel_importance_weight = model_param.channel_importance_weight
        self.percentile_quant = model_param.percentile_quant
        self.auto_entropy_model = model_param.auto_entropy_model
        if self.channel_importance_weight:
            logger.info("Improvement: channel_importance_weight ENABLED")
        if self.percentile_quant:
            logger.info("Improvement: percentile_quant ENABLED")
        if self.auto_entropy_model:
            logger.info("Improvement: auto_entropy_model ENABLED")

        logger.info("use config yaml")


        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._rotation = torch.empty(0)
        self._cov = torch.empty(0)
        self._euler = torch.empty(0)
        self._feature_indices = torch.empty(0)
        self.qas = nn.ModuleList([])
        self._V = None
        self.optimizer = None
        self.w = None
        self.val = None
        self.TMP = None
        self.res_tree = None
        self.ret_features = None
        
        # TopK SH 量化参数
        self._keep_q_indices = None
        self._keep_scales = None
        self._keep_zero_points = None
        self._keep_split = None
        self._num_keep = 0
        self._keep_mask = None
        self.setup_functions()
        self.n_sh = (self.max_sh_degree + 1) ** 2 
        logger.info(f'self.n_sh {self.n_sh}')
        if self.use_quat:
            self.n_rfc=11
        else:
            self.n_rfc=10
        
    
    def pre_volume(self, volume, beta):
        # volume = torch.tensor(volume)
        index = int(volume.shape[0] * 0.9)
        sorted_volume, _ = torch.sort(volume, descending=True)
        kth_percent_largest = sorted_volume[index]
        # Calculate v_list
        v_list = torch.pow(volume / kth_percent_largest, beta)
        return v_list

    def training_setup(self, training_args: MesonGSPlusOptimizationParams):
        logger.info(self.spatial_lr_scale)
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        
        if self.finetune_lr_scale < 1.0 - 0.001:
            logger.info('training setup: finetune')
            training_args.position_lr_init = training_args.position_lr_init * self.finetune_lr_scale
            training_args.feature_lr = training_args.feature_lr * self.finetune_lr_scale
            training_args.opacity_lr = training_args.opacity_lr * self.finetune_lr_scale
            training_args.scaling_lr = training_args.scaling_lr * self.finetune_lr_scale
            training_args.rotation_lr = training_args.rotation_lr * self.finetune_lr_scale
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    @task
    def cal_imp(
        self, 
        cam_iterator: CameraIterator, 
        pipe: PipelineParams,
        opt: MesonGSPlusOptimizationParams):
        """计算 importance 并立即剪枝（原始单 pruning rate 流程）。"""
        self.compute_imp(cam_iterator, pipe, opt)
        self.prune_mask()

    @torch.no_grad()
    def compute_imp(
        self, 
        cam_iterator: CameraIterator, 
        pipe: PipelineParams,
        opt: MesonGSPlusOptimizationParams):
        """
        只计算 importance（遍历所有训练视角渲染），不做剪枝。
        计算完成后 self.imp 包含所有点的 importance 值。
        可以之后调用 prune_mask() 或 prune_with_rate(rate) 做不同 rate 的剪枝。
        """
        beta_list = {
            'chair': 0.03,
            'drums': 0.05,
            'ficus': 0.03,
            'hotdog': 0.03,
            'lego': 0.05,
            'materials': 0.03,
            'mic': 0.03,
            'ship': 0.03,
            'bicycle': 0.03,
            'bonsai': 0.1,
            'counter': 0.1,
            'garden': 0.1,
            'kitchen': 0.1,
            'room': 0.1,
            'stump': 0.01,
        }   
        
        full_opa_imp = None
        bg_color = [1, 1, 1] if pipe.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=pipe.device)
        with torch.no_grad():
            for idx, view in enumerate(tqdm(cam_iterator, desc="count imp")):
                render_results = self.vanilla_render(
                    view, 
                    background,
                    pipe, 
                    opt, 
                    render_type='imp'
                )
                if full_opa_imp is None:
                    full_opa_imp = torch.zeros_like(render_results.imp).cuda()
                full_opa_imp.add_(render_results.imp)
                    
                del render_results
                gc.collect()
                torch.cuda.empty_cache()
            
        volume = torch.prod(self.scaling, dim=1)

        v_list = self.pre_volume(volume, beta_list.get(self.scene_imp, 0.1))
        imp = v_list * full_opa_imp
        
        self.imp = imp.detach()
        logger.info(f'compute_imp done: {self.imp.shape[0]} points, '
                    f'imp range [{self.imp.min().item():.4f}, {self.imp.max().item():.4f}]')
        
    @torch.no_grad
    def prune_mask(self):
        sorted_tensor, _ = torch.sort(self.imp, dim=0)
        index_nth_percentile = int(self.percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.imp <= value_nth_percentile).squeeze()
        self.imp = self.imp[torch.logical_not(prune_mask)]
        self.prune_points(prune_mask)
        logger.info(f'self finish pruning xyz {self.xyz.shape}')
    
    @torch.no_grad()
    def re_prune_and_rebuild_octree(self, target_pruning_rate: float):
        """
        Eval 阶段的二次剪枝：基于 checkpoint 中已有的 imp，按 target_pruning_rate 
        重新剪枝，并重建八叉树 + RAHT 参数 + VQ 索引映射。
        
        此方法在 encode() 开头调用，无需重新计算 imp（直接复用 checkpoint 中保存的 imp），
        也不重新训练 VQ codebook（复用已有 codebook，只对索引做裁剪）。
        
        Args:
            target_pruning_rate: 要剪掉的点的比例 [0.0, 1.0)。
                                 0.0 = 不剪枝；0.3 = 剪掉 30% 最不重要的点。
        """
        if target_pruning_rate <= 0.0:
            logger.info(f"re_prune_and_rebuild_octree: pruning_rate={target_pruning_rate}, skipping")
            return
        
        if target_pruning_rate >= 1.0:
            raise ValueError(f"pruning_rate must be < 1.0, got {target_pruning_rate}")
        
        # 检查 imp 是否可用
        if not hasattr(self, 'imp') or self.imp is None:
            logger.warning("re_prune_and_rebuild_octree: imp not available, cannot re-prune")
            return
        
        if self.imp.sum().item() == 0:
            logger.warning("re_prune_and_rebuild_octree: imp is all zeros (old checkpoint without imp), cannot re-prune")
            return
        
        n_before = self._xyz.shape[0]
        logger.info(f"re_prune_and_rebuild_octree: starting with {n_before} points, pruning_rate={target_pruning_rate}")
        
        # === Step 1: 基于 imp 计算剪枝 mask ===
        sorted_imp, _ = torch.sort(self.imp, dim=0)
        index_nth_percentile = int(target_pruning_rate * (sorted_imp.shape[0] - 1))
        value_nth_percentile = sorted_imp[index_nth_percentile]
        prune_mask = (self.imp <= value_nth_percentile).squeeze()
        keep_mask = ~prune_mask
        
        n_pruned = prune_mask.sum().item()
        n_keep = keep_mask.sum().item()
        logger.info(f"  Pruning {n_pruned}/{n_before} points ({n_pruned/n_before*100:.1f}%), keeping {n_keep}")
        
        # === Step 2: 过滤所有点属性 ===
        self._xyz = nn.Parameter(self._xyz.data[keep_mask].contiguous().requires_grad_(False))
        self._features_dc = nn.Parameter(self._features_dc.data[keep_mask].contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.data[keep_mask].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.data[keep_mask].contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.data[keep_mask].contiguous().requires_grad_(True))
        
        # 过滤 _features_rest：在 VQ 模式下，这是 codebook 不需要按点过滤；
        # 但 _feature_indices 需要按点过滤
        if self.use_indexed and self._feature_indices is not None and self._feature_indices.numel() > 0:
            # _feature_indices 是 per-point 的，需要过滤
            self._feature_indices = nn.Parameter(
                self._feature_indices.data[keep_mask].contiguous(), requires_grad=False)
            # _vq_indices_for_all 也是 per-point 的
            if hasattr(self, '_vq_indices_for_all') and self._vq_indices_for_all is not None:
                self._vq_indices_for_all = self._vq_indices_for_all[keep_mask].contiguous()
            # _keep_mask 也是 per-point 的
            if hasattr(self, '_keep_mask') and self._keep_mask is not None:
                self._keep_mask = self._keep_mask[keep_mask]
        else:
            # 非 indexed 模式，_features_rest 是 per-point 的
            self._features_rest = nn.Parameter(
                self._features_rest.data[keep_mask].contiguous().requires_grad_(True))
        
        # 过滤 imp
        self.imp = self.imp[keep_mask]
        
        # 过滤 auxiliary tensors
        if hasattr(self, 'max_radii2D') and self.max_radii2D is not None and self.max_radii2D.numel() == n_before:
            self.max_radii2D = self.max_radii2D[keep_mask]
        if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum is not None and self.xyz_gradient_accum.shape[0] == n_before:
            self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        if hasattr(self, 'denom') and self.denom is not None and self.denom.shape[0] == n_before:
            self.denom = self.denom[keep_mask]
        
        logger.info(f"  After pruning: xyz={self._xyz.shape}, imp={self.imp.shape}")
        
        # === Step 3: 重建八叉树 ===
        logger.info(f"  Rebuilding octree (depth={self.depth}, merge_type={self.merge_type})...")
        
        # 将 imp 添加到特征末尾作为第57维特征（与 octree_coding 一致）
        # 获取 per-point SH features（确保都是 [N, C] 2D tensor）
        if self.use_indexed:
            # indexed 模式: _features_rest 是 codebook [cb, 45]，通过 indices 索引得到 [N, 45]
            per_point_sh = self._features_rest[self._feature_indices.long()].detach().contiguous()
            if per_point_sh.dim() > 2:
                per_point_sh = per_point_sh.flatten(-2).contiguous()
        else:
            # 非 indexed 模式: _features_rest 是 [N, 15, 3]
            per_point_sh = self._features_rest.detach().flatten(-2).contiguous()
        
        features = torch.concat([
            self._opacity.detach(), 
            self._features_dc.detach().flatten(-2).contiguous(), 
            per_point_sh,
            self._scaling.detach(), 
            self._rotation.detach(),
            self.imp.detach().reshape(-1, 1)
        ], -1).cpu().numpy()
        
        V, features_merged, oct, paramarr, _, _ = create_octree_overall(
            self._xyz.detach().cpu().numpy(), 
            features,
            self.imp.cpu().numpy(),
            depth=self.depth,
            oct_merge=self.merge_type)
        dxyz, _ = decode_oct(paramarr, oct, self.depth)
        
        # 重建 RAHT 参数
        if self.raht:
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
            self.scale_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        
        # 解析合并后的特征
        opacities = features_merged[:, :1]
        features_dc = features_merged[:, 1:4].reshape(-1, 1, 3)
        features_extra = features_merged[:, 4:4 + 3 * (self.n_sh-1)].reshape(-1, self.n_sh - 1, 3)
        scales = features_merged[:, 49:52]
        rots = features_merged[:, 52:56]
        imp_merged = features_merged[:, 56:57]
        
        n_after_octree = dxyz.shape[0]
        
        # 更新模型参数
        self.oct = oct
        self.oct_param = paramarr
        self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.imp = torch.tensor(imp_merged, dtype=torch.float, device="cuda").squeeze()
        
        # === Step 4: 重建 VQ 索引映射 ===
        # 八叉树编码后点数可能减少（多点合并），需要重建 VQ 索引
        if self.use_indexed:
            # 八叉树合并后，features_extra 已是合并后的结果（per-point SH）
            # 需要对合并后的特征重新做 VQ 查找（使用已有 codebook）
            features_extra_tensor = torch.tensor(
                features_extra, dtype=torch.float, device="cuda"
            ).flatten(-2)  # [n_after_octree, 45]
            
            if hasattr(self, '_original_codebook') and self._original_codebook is not None:
                vq_codebook = self._original_codebook  # [cb_size, 45]
                
                # 为每个点找最近的 codebook entry
                # 使用 batch 计算避免 OOM
                n_points_new = features_extra_tensor.shape[0]
                new_vq_indices = torch.zeros(n_points_new, dtype=torch.long, device="cuda")
                batch_size = 50000
                for i in range(0, n_points_new, batch_size):
                    end = min(i + batch_size, n_points_new)
                    batch_feats = features_extra_tensor[i:end]  # [batch, 45]
                    # L2 距离查找最近 codebook entry
                    dists = torch.cdist(batch_feats.unsqueeze(0), vq_codebook.unsqueeze(0)).squeeze(0)  # [batch, cb_size]
                    new_vq_indices[i:end] = dists.argmin(dim=1)
                
                self._vq_indices_for_all = new_vq_indices
                self._feature_indices = nn.Parameter(new_vq_indices.contiguous(), requires_grad=False)
                
                # 重建 keep_mask 和 kept features
                # 基于 imp 重新选择 topk 保留点
                vq_cb_size = vq_codebook.shape[0]
                if self.sh_keep_topk > 0:
                    actual_k = min(self.sh_keep_topk, n_points_new)
                    _, topk_idx = torch.topk(self.imp, k=actual_k, largest=True)
                    new_keep_mask = torch.zeros(n_points_new, dtype=torch.bool, device="cuda")
                    new_keep_mask[topk_idx] = True
                    self._keep_mask = new_keep_mask
                    self._num_keep = actual_k
                    self._max_num_keep = actual_k
                    
                    # 提取 kept features 并构建新 codebook
                    kept_features = features_extra_tensor[new_keep_mask]
                    new_codebook = torch.cat([vq_codebook, kept_features], dim=0)
                    
                    # 更新 kept 点的索引（指向 codebook 中 kept 部分）
                    new_feature_indices = self._vq_indices_for_all.clone()
                    kept_global_indices = torch.where(new_keep_mask)[0]
                    new_feature_indices[kept_global_indices] = torch.arange(
                        vq_cb_size, vq_cb_size + actual_k,
                        dtype=torch.long, device="cuda"
                    )
                    self._feature_indices = nn.Parameter(new_feature_indices.contiguous(), requires_grad=False)
                    self._features_rest = nn.Parameter(new_codebook.contiguous(), requires_grad=True)
                    self._supports_dynamic_adjustment = True
                    logger.info(f"  VQ rebuilt: codebook=[{vq_cb_size} VQ + {actual_k} kept] = {new_codebook.shape[0]}")
                elif self.sh_keep_threshold > 0:
                    new_keep_mask = self.imp > self.sh_keep_threshold
                    actual_k = new_keep_mask.sum().item()
                    self._keep_mask = new_keep_mask
                    self._num_keep = actual_k
                    self._max_num_keep = actual_k
                    
                    if actual_k > 0:
                        kept_features = features_extra_tensor[new_keep_mask]
                        new_codebook = torch.cat([vq_codebook, kept_features], dim=0)
                        new_feature_indices = self._vq_indices_for_all.clone()
                        kept_global_indices = torch.where(new_keep_mask)[0]
                        new_feature_indices[kept_global_indices] = torch.arange(
                            vq_cb_size, vq_cb_size + actual_k,
                            dtype=torch.long, device="cuda"
                        )
                        self._feature_indices = nn.Parameter(new_feature_indices.contiguous(), requires_grad=False)
                        self._features_rest = nn.Parameter(new_codebook.contiguous(), requires_grad=True)
                    else:
                        self._features_rest = nn.Parameter(vq_codebook.contiguous(), requires_grad=True)
                    self._supports_dynamic_adjustment = True
                    logger.info(f"  VQ rebuilt: codebook=[{vq_cb_size} VQ + {actual_k} kept] = {self._features_rest.shape[0]}")
                else:
                    # 没有 topk/threshold，只用纯 VQ
                    self._features_rest = nn.Parameter(vq_codebook.contiguous(), requires_grad=True)
                    self._keep_mask = None
                    self._num_keep = 0
                    self._max_num_keep = 0
                    self._supports_dynamic_adjustment = True
                    logger.info(f"  VQ rebuilt: codebook=[{vq_cb_size} VQ only]")
            else:
                # 没有 VQ codebook，创建 identity mapping
                self._feature_indices = nn.Parameter(
                    torch.arange(n_after_octree, dtype=torch.long, device="cuda").contiguous(),
                    requires_grad=False)
                self._features_rest = nn.Parameter(
                    torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
                logger.info(f"  No VQ codebook, created identity mapping for {n_after_octree} points")
        else:
            # 非 indexed 模式
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        
        logger.info(f"re_prune_and_rebuild_octree complete: {n_before} -> {n_after_octree} points "
                    f"(pruned {n_before - n_after_octree}, {(n_before - n_after_octree)/n_before*100:.1f}%)")
    
    @task
    @torch.no_grad
    def octree_coding(self):
        # 记录八叉树编码前的 importance 统计
        logger.info(f"Importance before octree: min={self.imp.min().item():.4f}, max={self.imp.max().item():.4f}, mean={self.imp.mean().item():.4f}")
        
        # 将 imp 添加到特征末尾作为第57维特征
        features = torch.concat([
            self._opacity.detach(), 
            self._features_dc.detach().flatten(-2).contiguous(), 
            self._features_rest.detach().flatten(-2).contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach(),
            self.imp.detach().reshape(-1, 1)  # 新增：将 importance 作为特征
        ], -1).cpu().numpy()

        V, features, oct, paramarr, _, _ = create_octree_overall(
            self._xyz.detach().cpu().numpy(), 
            features,
            self.imp.cpu().numpy(),
            depth=self.depth,
            oct_merge=self.merge_type)
        dxyz, _ = decode_oct(paramarr, oct, self.depth)
        
        if self.raht:
            # morton sort
            logger.info("here raht?")
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
            self.scale_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        
        opacities = features[:, :1]
        features_dc = features[:, 1:4].reshape(-1, 1, 3)
        features_extra = features[:, 4:4 + 3 * (self.n_sh-1)].reshape(-1, self.n_sh - 1, 3)
        scales = features[:, 49:52]
        rots = features[:, 52:56]
        imp_merged = features[:, 56:57]  # 新增：提取合并后的 importance
        
        self.oct = oct
        self.oct_param = paramarr
        self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 新增：更新 importance 以与点对齐
        self.imp = torch.tensor(imp_merged, dtype=torch.float, device="cuda").squeeze()
        logger.info(f"Importance preserved through octree_coding: {self.imp.shape}")
        logger.info(f"Importance stats after octree: min={self.imp.min().item():.4f}, max={self.imp.max().item():.4f}, mean={self.imp.mean().item():.4f}")
        
        # 如果使用 indexed 模式但没有执行 vq_fe，创建 identity mapping
        if self.use_indexed and (self._feature_indices.numel() == 0 or self._feature_indices.dtype != torch.long):
            n_points = self._xyz.shape[0]
            self._feature_indices = nn.Parameter(
                torch.arange(n_points, dtype=torch.long, device="cuda").contiguous(), 
                requires_grad=False
            )
            logger.info(f"Created identity feature_indices for indexed mode: {self._feature_indices.shape}")
        
        print("finish octree coding")

    @task
    def init_qas(self):
        n_qs = self.n_rfc * self.n_block
        for i in range(n_qs): 
            self.qas.append(VanillaQuan(bit=self.num_bits).cuda())
        logger.info(f'Init qa, length: {n_qs}')
    
    @task
    @torch.no_grad
    def vq_fe(self):
        print("start vq_fe")
        features_extra = self._features_rest.detach().flatten(-2)
        
        # imp 已经通过 octree_coding 对齐，直接使用
        logger.info(f"Using importance for VQ: shape={self.imp.shape}, topk={self.sh_keep_topk}, threshold={self.sh_keep_threshold}")
        
        # 如果设置了 top-k，显示会保留多少点
        if self.sh_keep_topk > 0:
            actual_k = min(self.sh_keep_topk, self.imp.shape[0])
            logger.info(f"Will keep top-{actual_k} most important points with full SH")
        # 如果设置了阈值，显示会保留多少点（使用原始importance值，不归一化）
        elif self.sh_keep_threshold > 0:
            num_keep = (self.imp > self.sh_keep_threshold).sum().item()
            logger.info(f"Will keep {num_keep}/{self.imp.shape[0]} points with full SH (raw importance > {self.sh_keep_threshold})")
        
        codebook, vq_indices, quant_info = vq_features(
            features_extra,
            self.imp,
            self.codebook_size,
            self.batch_size,
            self.steps,
            sh_keep_threshold=self.sh_keep_threshold,
            sh_keep_topk=self.sh_keep_topk,
            quantize_kept=True,
            kept_quant_bits=8,
            kept_quant_blocks=self.n_block,
        )

        self._feature_indices = nn.Parameter(vq_indices.detach().contiguous(), requires_grad=False)
        self._features_rest = nn.Parameter(codebook.detach().contiguous(), requires_grad=True)
        
        # 存储量化参数
        self._keep_q_indices = quant_info['keep_q_indices']
        self._keep_scales = quant_info['keep_scales']
        self._keep_zero_points = quant_info['keep_zero_points']
        self._keep_split = quant_info['keep_split']
        self._num_keep = quant_info['num_keep']
        self._keep_mask = quant_info['keep_mask']

        # 保存原始的VQ索引映射（用于eval阶段的动态调整）
        # _vq_indices_for_all: 所有点的VQ索引（包括kept点的VQ索引！）
        # 这个索引指向纯VQ codebook，所有索引都在[0, vq_cb_size)范围内
        self._vq_indices_for_all = quant_info['vq_indices_for_all'].detach().clone()
        # _original_codebook: 纯VQ codebook（不包含kept点！）
        # 这是所有点做VQ的结果，永远不会被修改
        self._original_codebook = quant_info['pure_vq_codebook'].detach().clone()
        # _max_num_keep: train阶段设置的最大保留点数量
        self._max_num_keep = self._num_keep
        # _supports_dynamic_adjustment: 明确标记是否支持动态调整
        self._supports_dynamic_adjustment = True

        vq_cb_size = self._original_codebook.shape[0]
        logger.info(f"VQ completed: num_keep={self._num_keep}, max_num_keep={self._max_num_keep}, "
                   f"vq_cb_size={vq_cb_size}, quantized={self._keep_q_indices is not None}, "
                   f"supports_dynamic_adjust={self._supports_dynamic_adjustment}")
    
    @property
    def original_rotation(self):
        return self._rotation
    
    @property
    def original_opacity(self):
        return self._opacity
    
    @property
    def original_scales(self):
        return self._scaling
    
    @property
    def get_features_extra(self):
        features_extra = self._features_rest.reshape((-1, 3, (self.max_sh_degree + 1) ** 2 - 1))
        return features_extra
    
    @property
    def feature_indices(self):
        return self._feature_indices
    
    @property
    def get_indexed_feature_extra(self):
        n_sh = (self.active_sh_degree + 1) ** 2
        num_points = self.xyz.shape[0]
        fi = self._feature_indices.detach().cpu()
        fr = self._features_rest.detach().cpu()
        ret = torch.zeros([num_points, 3 * (n_sh - 1)])
        for i in range(num_points):
            ret[i] = self._features_rest[int(fi[i])]
        return ret.reshape(-1, n_sh - 1, 3)
        # return torch.matmul(F.one_hot(self._feature_indices).float(), self._features_rest).reshape(-1, self.n_sh - 1, 3)

    @property
    def get_cov(self):
        return self._cov

    @property
    def get_euler(self):
        return self._euler
    
    def get_covariance(self, scaling_modifier=1):
        if self.get_euler.shape[0] > 0 and (not self.use_quat):
            # print('go with euler')
            return self.covariance_activation_for_euler(self.scaling, scaling_modifier, self._euler)
        elif self.get_cov.shape[0] > 0:
            return self.get_cov
        else:
            # print('gaussian model: get cov from scaling and rotations.')
            return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)
        
    def capture(self):
        supports_dynamic_adjustment = getattr(self, '_supports_dynamic_adjustment', False)
        logger.info(f"Capturing checkpoint: supports_dynamic_adjustment={supports_dynamic_adjustment}, "
                   f"max_num_keep={getattr(self, '_max_num_keep', None)}, num_keep={self._num_keep}")
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._cov,
            self._euler,
            self._feature_indices,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.qas.state_dict(),
            self.reorder,
            self.res,
            self.oct,
            self.oct_param,
            self._keep_q_indices,
            self._keep_scales,
            self._keep_zero_points,
            self._keep_split,
            self._num_keep,
            self._keep_mask,
            self._vq_indices_for_all,
            self._max_num_keep,
            self._original_codebook,
            supports_dynamic_adjustment,
            getattr(self, 'imp', None),  # Save imp attribute
        )

    def restore(self, model_args, training_args=None):
        # 灵活处理不同版本的checkpoint (backward compatibility)
        num_args = len(model_args)
        logger.info(f"Loading checkpoint with {num_args} elements")

        # 根据checkpoint中的元素数量判断版本
        if num_args >= 31:
            # 最新格式 (31个元素): 包含动态调整字段 + supports_dynamic_adjustment标志 + imp
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._cov,
                self._euler,
                self._feature_indices,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                qas_state_dict,
                self.reorder,
                self.res,
                self.oct,
                self.oct_param,
                _keep_q_indices,
                _keep_scales,
                _keep_zero_points,
                _keep_split,
                _num_keep,
                _keep_mask,
                _vq_indices_for_all,
                _max_num_keep,
                _original_codebook,
                _supports_dynamic_adjustment,
                imp,
            ) = model_args
            self.imp = imp if imp is not None else torch.zeros(self._xyz.shape[0], device="cuda")
        elif num_args >= 30:
            # 新格式 (30个元素): 包含动态调整所需的全部字段 + supports_dynamic_adjustment标志
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._cov,
                self._euler,
                self._feature_indices,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                qas_state_dict,
                self.reorder,
                self.res,
                self.oct,
                self.oct_param,
                _keep_q_indices,
                _keep_scales,
                _keep_zero_points,
                _keep_split,
                _num_keep,
                _keep_mask,
                _vq_indices_for_all,
                _max_num_keep,
                _original_codebook,
                _supports_dynamic_adjustment,
            ) = model_args
            # 为旧格式初始化imp（全零，表示没有importance信息）
            self.imp = torch.zeros(self._xyz.shape[0], device="cuda")
        elif num_args >= 29:
            # 旧的新格式 (29个元素): 包含动态调整字段但缺少supports_dynamic_adjustment标志
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._cov,
                self._euler,
                self._feature_indices,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                qas_state_dict,
                self.reorder,
                self.res,
                self.oct,
                self.oct_param,
                _keep_q_indices,
                _keep_scales,
                _keep_zero_points,
                _keep_split,
                _num_keep,
                _keep_mask,
                _vq_indices_for_all,
                _max_num_keep,
                _original_codebook,
            ) = model_args
            # 设置标志为True（因为29元素的格式本来就是为了支持动态调整）
            _supports_dynamic_adjustment = True
            # 为旧格式初始化imp
            self.imp = torch.zeros(self._xyz.shape[0], device="cuda")
        elif num_args >= 26:
            # 中间格式 (26-28个元素): 包含TopK字段，但不包含动态调整字段
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._cov,
                self._euler,
                self._feature_indices,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                qas_state_dict,
                self.reorder,
                self.res,
                self.oct,
                self.oct_param,
                _keep_q_indices,
                _keep_scales,
                _keep_zero_points,
                _keep_split,
                _num_keep,
                _keep_mask,
            ) = model_args[:26]
            # 为中间格式初始化动态调整字段
            # 从self._feature_indices和self._features_rest重建动态调整所需的信息
            if self._feature_indices is not None and self._features_rest is not None:
                # _vq_indices_for_all是最终的索引（指向原始codebook）
                _vq_indices_for_all = self._feature_indices.detach().clone()
                # _original_codebook是当前的codebook（已经是kept + VQ的组合）
                _original_codebook = self._features_rest.detach().clone()
                # _max_num_keep是原始kept points的数量（用于调整时的参考）
                # 可以从num_keep推断，或者设置为0表示不支持动态调整
                _max_num_keep = _num_keep if _num_keep is not None else 0
                logger.info(f"Restored intermediate format: num_keep={_num_keep}, max_num_keep={_max_num_keep}, codebook_size={_original_codebook.shape[0]}")
            else:
                _vq_indices_for_all = None
                _original_codebook = None
                _max_num_keep = _num_keep if _num_keep is not None else 0
            # 中间格式不支持动态调整
            _supports_dynamic_adjustment = False
            # 为旧格式初始化imp
            self.imp = torch.zeros(self._xyz.shape[0], device="cuda")
        else:
            # 旧格式 (20个元素): 不包含TopK字段
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._cov,
                self._euler,
                self._feature_indices,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                qas_state_dict,
                self.reorder,
                self.res,
                self.oct,
                self.oct_param,
            ) = model_args[:20]
            # 为旧格式初始化TopK和动态调整字段
            _keep_q_indices = None
            _keep_scales = None
            _keep_zero_points = None
            _keep_split = None
            _num_keep = 0
            _keep_mask = None
            _original_codebook = None
            _vq_indices_for_all = None
            _max_num_keep = 0
            _supports_dynamic_adjustment = False
            # 为旧格式初始化imp
            self.imp = torch.zeros(self._xyz.shape[0], device="cuda")

        if training_args is not None:
            self.training_setup(training_args)

        # Since training_setup will reset these parameters, we assign values to them manually
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self._keep_q_indices = _keep_q_indices
        self._keep_scales = _keep_scales
        self._keep_zero_points = _keep_zero_points
        self._keep_split = _keep_split
        self._num_keep = _num_keep
        self._keep_mask = _keep_mask
        self._vq_indices_for_all = _vq_indices_for_all
        self._max_num_keep = _max_num_keep
        self._original_codebook = _original_codebook
        self._supports_dynamic_adjustment = _supports_dynamic_adjustment
        logger.info(f"Checkpoint restored: supports_dynamic_adjustment={self._supports_dynamic_adjustment}, "
                   f"max_num_keep={self._max_num_keep}, num_keep={self._num_keep}")
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        # n_qs = self.n_rfc * self.n_block
        # for i in range(n_qs):
        #     self.qas.append(VanillaQuan(bit=self.num_bits).cuda())
        #self.qas.load_state_dict(qas_state_dict)


    def vanilla_render(
        self,
        viewpoint_camera, 
        background,
        pipe: PipelineParams,
        opt: MesonGSPlusOptimizationParams,
        scaling_modifier: float=1.0,
        override_color=None,
        render_type: str = 'imp'
    ):
        meson_count = False
        if render_type == 'imp':
            meson_count = True
        
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            clamp_color=self.clamp_color,
            meson_count=meson_count,
            f_count=False,
            depth_count=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
        means3D = self.xyz
        means2D = screenspace_points

        opacity = self.opacity
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.use_quat:
            scales = self.scaling
            rotations = self.rotation
        elif pipe.compute_cov3D_python or self.get_cov.shape[0] > 0 or self.get_euler.shape[0] > 0:
            cov3D_precomp = self.get_covariance(scaling_modifier)
            # print('gaussian_renderer __init__', cov3D_precomp.shape)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = self.features.transpose(1, 2).view(-1, 3, (self.active_sh_degree+1)**2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # print('shs_view', shs_view.max(), shs_view.min())
                # print('active_sh_degree', pc.active_sh_degree)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                # print('sh2rgb.max(), sh2rgb.min()', sh2rgb.max(), sh2rgb.min())
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # colors_precomp = colors_precomp.nan_to_num(0)
                # colors_precomp = 
                # print('colors_precomp', colors_precomp.max(), colors_precomp.min(), colors_precomp[:5])
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        if meson_count:
            rendered_image, radii, imp = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            return MesonGSPlusRenderResult(
                    rendered_image=rendered_image,
                    viewspace_points=screenspace_points,
                    visibility_filter=radii > 0,
                    radii=radii,
                    imp=imp
                )
        else:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            #print("radii", radii.shape) #[2451926]
            return MesonGSPlusRenderResult(
                    rendered_image=rendered_image,
                    viewspace_points=screenspace_points,
                    visibility_filter=radii > 0,
                    radii=radii
                )
        # elif f_count:
        #     rendered_image, radii, imp, gaussians_count, opa_imp = rasterizer(
        #         means3D = means3D,
        #         means2D = means2D,
        #         shs = shs,
        #         colors_precomp = colors_precomp,
        #         opacities = opacity,
        #         scales = scales,
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp)
        #     return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter" : radii > 0,
        #         "radii": radii,
        #         "imp": imp,
        #         "gaussians_count": gaussians_count,
        #         "opa_imp": opa_imp}

        # elif depth_count:
        #     rendered_image, radii, out_depth = rasterizer(
        #         means3D = means3D,
        #         means2D = means2D,
        #         shs = shs,
        #         colors_precomp = colors_precomp,
        #         opacities = opacity,
        #         scales = scales,
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp)
        #     return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter" : radii > 0,
        #         "radii": radii,
        #         "depth": out_depth}
        
        
        
    def render(
        self, 
        viewpoint_camera, 
        background,
        pipe: PipelineParams,
        opt: MesonGSPlusOptimizationParams = None,
        step: int = None,
        scaling_modifier: float=1.0,
        override_color=None,
        render_type: str ='ft'): 
        # logger.info(f'recent step {step}')
        if (render_type == 'vanilla') or (pipe.eval_mode is not None): #当前我想要知道预训练的渲染效果上限，下面实现了初步地seg量化
            return self.vanilla_render(
                viewpoint_camera,
                background,
                pipe,
                opt,
                scaling_modifier,
                override_color,
                render_type
            )
        print("start quanting eval")
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            clamp_color=self.clamp_color,
            meson_count=False,
            f_count=False,
            depth_count=False
        )
        
        # 根据 use_indexed 选择不同的 rasterizer
        if self.use_indexed:
            rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings)
        else:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        if self.use_quat:
            re_range = [1, 5]
            shzero_range = [5, 8]
        else:
            re_range = [1, 4]
            shzero_range = [4, 7]
        
        means3D = self.xyz
        means2D = screenspace_points
        
        if self.raht:
            if self.use_quat:
                rf = torch.concat([self.original_opacity, self.original_rotation, self.features_dc.contiguous().squeeze()], -1)
            else:
                r = self.original_rotation
                norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                q = r / norm[:, None]
                eulers = ToEulerAngles_FT(q)
                rf = torch.concat([self.original_opacity, eulers, self.features_dc.contiguous().squeeze()], -1)

            C = rf[self.reorder]
            iW1 = self.res['iW1']
            iW2 = self.res['iW2']
            iLeft_idx = self.res['iLeft_idx']
            iRight_idx = self.res['iRight_idx']

            for d in range(self.depth * 3):
                w1 = iW1[d]
                w2 = iW2[d]
                left_idx = iLeft_idx[d]
                right_idx = iRight_idx[d]
                C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                    w2, 
                                                    C[left_idx], 
                                                    C[right_idx])
            
            quantC = torch.zeros_like(C)
            quantC[0] = C[0]
            # if self.per_channel_quant:
            #     for i in range(C.shape[-1]):
            #         quantC[1:, i] = self.qas[i](C[1:, i])
            # elif self.per_block_quant:
            # ==================== SegQuant Replacement for quantC ====================
            lc1 = C.shape[0] - 1
            channels = C.shape[-1]
            split_ac = split_length(lc1, self.n_block)
            
            splits_tensor = torch.tensor(split_ac, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels, 1)
            power_qbits_tensor = torch.full((channels, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)

            q_indices, scales, zps, _, _ = pure_quant_wo_minmax(C[1:], splits_tensor, power_qbits_tensor)
            
            start = 0
            q_outputs_list = []
            for b in range(self.n_block):
                length = split_ac[b]
                block_q = q_indices[:, start:start+length] 
                block_scale = scales[:, b:b+1]
                block_zp = zps[:, b:b+1]
                
                block_deq = block_scale * (block_q.float() - block_zp)
                q_outputs_list.append(block_deq)
                start += length
                
            quantC_transposed = torch.cat(q_outputs_list, dim=1)
            quantC[1:] = quantC_transposed.transpose(0, 1)
            # ==================== SegQuant Replacement End ====================
            
            # else:
            #     quantC[1:] = self.qa(C[1:])

            res_inv = self.res_inv
            pos = res_inv['pos']
            iW1 = res_inv['iW1']
            iW2 = res_inv['iW2']
            iS = res_inv['iS']
            
            iLeft_idx = res_inv['iLeft_idx']
            iRight_idx = res_inv['iRight_idx']
        
            iLeft_idx_CT = res_inv['iLeft_idx_CT']
            iRight_idx_CT = res_inv['iRight_idx_CT']
            iTrans_idx = res_inv['iTrans_idx']
            iTrans_idx_CT = res_inv['iTrans_idx_CT'] 

            CT_yuv_q_temp = quantC[pos.astype(int)]
            raht_features = torch.zeros(quantC.shape).cuda()
            OC = torch.zeros(quantC.shape).cuda()
            
            for i in range(self.depth*3):
                w1 = iW1[i]
                w2 = iW2[i]
                S = iS[i]
                
                left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
                left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
                
                trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
                
                
                OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
                OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                        w2, 
                                                        CT_yuv_q_temp[left_idx_CT], 
                                                        CT_yuv_q_temp[right_idx_CT])  
                CT_yuv_q_temp[:S] = OC[:S]

            raht_features[self.reorder] = OC
            
            scales = self.original_scales
            

            # ==================== SegQuant Scales Replacement ====================
            lc_scale = scales.shape[0]
            channels_scale = scales.shape[-1]
            split_scale = split_length(lc_scale, self.n_block)
            
            splits_tensor_scale = torch.tensor(split_scale, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels_scale, 1)
            power_qbits_scale = torch.full((channels_scale, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)
            
            q_indices_s, scales_s, zps_s, _, _ = pure_quant_wo_minmax(scales, splits_tensor_scale, power_qbits_scale)
            
            start = 0
            s_outputs_list = []
            for b in range(self.n_block):
                length = split_scale[b]
                block_q = q_indices_s[:, start:start+length]
                block_scale = scales_s[:, b:b+1]
                block_zp = zps_s[:, b:b+1]
                block_deq = block_scale * (block_q.float() - block_zp)
                s_outputs_list.append(block_deq)
                start += length
            
            scalesq_transposed = torch.cat(s_outputs_list, dim=1)
            scalesq = scalesq_transposed.transpose(0, 1)
            # ==================== SegQuant Scales Replacement End ====================
                    
            scaling = torch.exp(scalesq)
            
            if self.use_quat:
                rotations = raht_features[:, 1:5]
                cov3D_precomp = self.covariance_activation(scaling, 1.0, rotations)
            else:
                eulers = raht_features[:, 1:4]
                cov3D_precomp = self.covariance_activation_for_euler(scaling, 1.0, eulers)

            assert cov3D_precomp is not None
            
            opacity = raht_features[:, :1]
            opacity = torch.sigmoid(opacity)    
            
            scales = None
            rotations = None
            eulers = None
            colors_precomp = None
            
            if self.use_indexed:
                sh_zero = raht_features[:, shzero_range[0]:].unsqueeze(1).contiguous()
                sh_ones = self.get_features_extra.reshape(-1, (self.active_sh_degree+1)**2 - 1, 3)
                sh_indices = self.feature_indices
            else:
                # 非 indexed 模式：直接使用 _features_rest，不通过 codebook 索引
                features_dc = raht_features[:, shzero_range[0]:].unsqueeze(1)
                feature_extra = self._features_rest
                features = torch.cat((features_dc, feature_extra), dim=1)
                shs_view = features.transpose(1, 2).view(-1, 3, (self.active_sh_degree+1)**2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # 初始化 indexed 模式的变量为 None
                sh_zero = None
                sh_ones = None
                sh_indices = None
        else:
            raise Exception("Sorry, w/o raht version is unimplemented.")
        
        # 根据 use_indexed 选择不同的调用方式
        if self.use_indexed:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                opacities = opacity,
                sh_indices = sh_indices,
                sh_zero = sh_zero,
                sh_ones = sh_ones,
                colors_precomp = colors_precomp,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        else:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                opacities = opacity,
                shs = None,
                colors_precomp = colors_precomp,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

        return MesonGSPlusRenderResult(
                rendered_image=rendered_image,
                viewspace_points=screenspace_points,
                visibility_filter=radii > 0,
                radii=radii
            )
    
    @task
    def finetuning_setup(self, training_args: MesonGSPlusOptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
    
        print('finetuning setup: finetune')
        training_args.position_lr_init = training_args.position_lr_init * self.finetune_lr_scale
        training_args.feature_lr = training_args.feature_lr * self.finetune_lr_scale
        training_args.opacity_lr = training_args.opacity_lr * self.finetune_lr_scale
        training_args.scaling_lr = training_args.scaling_lr * self.finetune_lr_scale
        training_args.rotation_lr = training_args.rotation_lr * self.finetune_lr_scale * 0.2
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    @task
    def update_learning_rate(self, iteration: int):
        iteration += 30000 # meson customize
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: MesonGSPlusOptimizationParams):

        # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)
        # scheduler.register_task(1, task=self.calc_importance_task)
        # scheduler.register_task(1, task=self.training_setup)
        logger.info('registering tasks')
        scheduler.register_task(1, task=self.cal_imp)
        scheduler.register_task(1, task=self.octree_coding)
        if self.use_indexed:
            scheduler.register_task(1, task=self.vq_fe) # 球谐函数的量化
        #scheduler.register_task(1, task=self.init_qas)
        #scheduler.register_task(1, task=self.finetuning_setup)
        #scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        # scheduler.register_task(1, task=self.re_exec_training_setup, priority=1) # exec after vq_compress
        # pass
    
    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: MesonGSPlusOptimizationParams):
        pass

    @torch.no_grad
    def _adjust_num_keep_for_size_limit(self):
        """
        根据size_limit动态调整保留点数量。

        设计思路（Budget-First 策略）：
        1. 先以 num_keep=0 做一次 save_npz（使用最高量化位宽），得到基础压缩大小 base_size
           — 这保证了 RAHT 量化部分使用了最佳的 bit budget
        2. 计算 remaining_budget = size_limit - base_size
        3. 通过小样本测试估算每个 kept point 的 Laplace 编码后平均字节数
        4. 把 remaining_budget 全部分配给 kept points: num_keep = remaining_budget / per_keep_bytes
        5. 后续 encode 中的 search_qbits 会在加入 kept points 后自动重新分配 RAHT bit budget

        这样的好处：
        - 不需要手动指定 num_keep，完全由 size_limit 自动决定
        - 保证了基础压缩的质量底线（RAHT 部分用最好的 bit width）
        - 在质量底线之上，尽可能多地保留原始 SH 数据

        Returns:
            adjusted_cb: 调整后的codebook
            adjusted_indices: 调整后的feature_indices
            adjusted_num_keep: 调整后的保留点数量
            adjusted_q_indices: 调整后的量化索引 (None)
            adjusted_scales: 调整后的scale (None)
            adjusted_zero_points: 调整后的zero_point (None)
            adjusted_split: 调整后的分块信息 (None)
        """
        # 检查是否支持动态调整
        is_new_format = getattr(self, '_supports_dynamic_adjustment', False)

        if not is_new_format:
            logger.info(f"Dynamic adjustment not available, using original codebook")
            return (self._features_rest, self._feature_indices, self._num_keep,
                    self._keep_q_indices, self._keep_scales, self._keep_zero_points, self._keep_split)

        _max_num_keep = self._max_num_keep
        size_limit_bytes = self.size_limit_mb * 1024 * 1024

        if size_limit_bytes <= 0:
            logger.info("size_limit_mb <= 0, using all kept points")
            return (self._features_rest, self._feature_indices, self._num_keep,
                    self._keep_q_indices, self._keep_scales, self._keep_zero_points, self._keep_split)

        logger.info(f"=== Budget-First num_keep allocation ===")
        logger.info(f"size_limit={self.size_limit_mb}MB, max_num_keep={_max_num_keep}")

        # --- Step 1: 估算 num_keep=0 时的 codebook 大小 ---
        # 只测量 codebook 部分（不做完整 save_npz 以避免重复 RAHT 变换）
        import tempfile, os
        from .meson_utils import quantize_kept_sh
        from .laplace_codec import laplace_encode_blocks

        (base_cb_tensor, _, _) = self._adjust_num_keep_with_value(0)
        base_cb_np = base_cb_tensor.detach().contiguous().cpu().numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez_compressed(os.path.join(tmpdir, 'um.npz'), umap=base_cb_np)
            base_cb_bytes = os.path.getsize(os.path.join(tmpdir, 'um.npz'))

        logger.info(f"Step 1 - Base CB (num_keep=0, npz): {base_cb_bytes/1024:.1f}KB")

        # --- Step 2: 用两个不同 num_keep 值的 Laplace 编码差来估算 per_keep_bytes ---
        # 关键：num_keep=0 用 npz_compressed 格式，num_keep>0 用 Laplace 格式
        # 比较两种不同格式会产生负 delta。所以用两个 Laplace 编码的差值。
        probe_small = min(5000, _max_num_keep)
        probe_large = min(50000, _max_num_keep)
        if probe_small <= 0 or probe_large <= probe_small:
            logger.info("No kept points available or too few for probe")
            return self._adjust_num_keep_with_value(0) + (None, None, None, None)

        def _measure_laplace_cb_bytes(num_keep_val):
            """测量 Laplace 编码后的 codebook 大小"""
            (cb_tensor, _, _) = self._adjust_num_keep_with_value(num_keep_val)
            cb_torch = cb_tensor.detach().contiguous().cuda()
            q_indices, scales_q, zps_q, split_q = quantize_kept_sh(
                cb_torch, n_block=self.cb_n_block, num_bits=self.cb_quant_bits
            )
            q_np = q_indices.detach().cpu().numpy()
            scales_np = scales_q.detach().cpu().numpy()
            zps_np = zps_q.detach().cpu().numpy()
            bitstream = laplace_encode_blocks(q_np, split_q, q_np.shape[0])
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'cb_q.bin'), 'wb') as f:
                    f.write(bitstream)
                np.savez_compressed(os.path.join(tmpdir, 'cb_meta.npz'),
                    scales=scales_np, zero_points=zps_np,
                    split=np.array(split_q),
                    num_entries=cb_torch.shape[0],
                    num_keep=num_keep_val)
                cb_bytes = os.path.getsize(os.path.join(tmpdir, 'cb_q.bin')) + \
                           os.path.getsize(os.path.join(tmpdir, 'cb_meta.npz'))
            del q_indices, scales_q, zps_q, cb_torch
            torch.cuda.empty_cache()
            return cb_bytes

        small_cb_bytes = _measure_laplace_cb_bytes(probe_small)
        large_cb_bytes = _measure_laplace_cb_bytes(probe_large)
        
        cb_delta_bytes = large_cb_bytes - small_cb_bytes
        delta_num_keep = probe_large - probe_small
        per_keep_bytes = cb_delta_bytes / delta_num_keep if delta_num_keep > 0 else 1.0

        # 安全下限：不低于 1 byte/point（Laplace 编码不可能比 1 byte 更小）
        if per_keep_bytes <= 1.0:
            # 理论下限：量化后每个点至少 cb_quant_bits 位每个通道，Laplace 压缩约 50-70%
            cb_dim = self._original_codebook.shape[1]
            per_keep_raw_bytes = cb_dim * (self.cb_quant_bits / 8.0)
            per_keep_bytes = max(per_keep_bytes, per_keep_raw_bytes * 0.3)
            logger.warning(f"per_keep_bytes={cb_delta_bytes/delta_num_keep:.2f} too low, "
                          f"clamped to {per_keep_bytes:.2f}")

        logger.info(f"Step 2 - Codebook probe (Laplace-Laplace comparison):")
        logger.info(f"  Small({probe_small}): {small_cb_bytes/1024:.1f}KB, "
                   f"Large({probe_large}): {large_cb_bytes/1024:.1f}KB")
        logger.info(f"  Delta: {cb_delta_bytes/1024:.1f}KB / {delta_num_keep} pts "
                   f"→ per_keep={per_keep_bytes:.2f} bytes/point")

        # --- Step 3: 计算可用给 kept points 的空间 ---
        # 编码总大小 = fixed_overhead (oct+ntk) + codebook + RAHT(orgb+ct) + metadata
        # 其中 RAHT 由 search_qbits 自动分配，codebook 由 num_keep 决定
        # search_qbits 机制：other_bytes = oct + ntk + cb 先扣除，剩余空间给 RAHT ILP
        # 当 cb 变大时，RAHT budget 自动缩小，ILP 用更低 bit width
        #
        # 策略：oct + ntk + base_cb 是固定开销，给 RAHT 预留合理比例
        # 剩余空间全给 kept points 的 codebook 增量
        n_points = self._feature_indices.shape[0]
        base_ntk = self._adjust_num_keep_with_value(0)[1].detach().contiguous().cpu().int().numpy()
        # Use GPCC for octree, Laplace for NTK
        if is_gpcc_available():
            try:
                oct_bitstream = gpcc_encode_octree(self.oct, self.oct_param, self.depth)
                oct_bytes = len(oct_bitstream)
                logger.info(f"GPCC octree overhead estimation: {oct_bytes/1024:.1f}KB")
            except Exception as e:
                logger.warning(f"GPCC encode failed in overhead estimation, falling back to zlib: {e}")
                with tempfile.TemporaryDirectory() as tmpdir:
                    np.savez_compressed(os.path.join(tmpdir, 'oct.npz'),
                                        points=self.oct, params=self.oct_param)
                    oct_bytes = os.path.getsize(os.path.join(tmpdir, 'oct.npz'))
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                np.savez_compressed(os.path.join(tmpdir, 'oct.npz'),
                                    points=self.oct, params=self.oct_param)
                oct_bytes = os.path.getsize(os.path.join(tmpdir, 'oct.npz'))
        ntk_bitstream = ntk_encode(base_ntk, n_block=self.ntk_n_block)
        ntk_bytes = len(ntk_bitstream)
        logger.info(f"Laplace NTK overhead estimation: {ntk_bytes/1024:.1f}KB (n_block={self.ntk_n_block})")

        fixed_overhead_bytes = oct_bytes + ntk_bytes + base_cb_bytes
        logger.info(f"Step 3 - Fixed overhead: oct={oct_bytes/1024:.1f}KB + ntk={ntk_bytes/1024:.1f}KB "
                    f"+ baseCB={base_cb_bytes/1024:.1f}KB = {fixed_overhead_bytes/1024/1024:.2f}MB")

        # RAHT 预留比例：根据 fixed_overhead 在 size_limit 中的占比自适应
        # 核心思想：空间紧张时 RAHT 质量是主要瓶颈，需要更多空间
        # 空间充裕时 kept points 的边际收益更高
        overhead_ratio = fixed_overhead_bytes / size_limit_bytes
        if overhead_ratio > 0.5:
            # 空间非常紧张（固定开销 > 50%），大部分空间留给 RAHT
            raht_reserve_ratio = 0.50
        elif overhead_ratio > 0.35:
            # 空间较紧张，RAHT 预留更多
            raht_reserve_ratio = 0.45
        else:
            # 空间充裕，可以给 kept points 更多
            raht_reserve_ratio = 0.40

        remaining_bytes = size_limit_bytes * (1.0 - raht_reserve_ratio) - fixed_overhead_bytes
        logger.info(f"  overhead_ratio={overhead_ratio:.2f}, raht_reserve={raht_reserve_ratio}")
        logger.info(f"  Remaining for kept points: {remaining_bytes/1024/1024:.2f}MB")

        if remaining_bytes <= 0:
            logger.info(f"No budget for kept points")
            return self._adjust_num_keep_with_value(0) + (None, None, None, None)

        # --- Step 4: 分配 kept points ---
        # 安全系数 0.85：codebook 增大后 search_qbits 会自动调低 RAHT bit width，
        # 实际总 size 应仍在 size_limit 内（search_qbits 有多轮迭代保证）
        safety_factor = 0.85
        estimated_num_keep = int(remaining_bytes * safety_factor / per_keep_bytes)
        new_num_keep = max(0, min(estimated_num_keep, _max_num_keep))

        logger.info(f"Step 4 - Budget allocation:")
        logger.info(f"  remaining={remaining_bytes/1024/1024:.2f}MB × safety={safety_factor} "
                    f"/ per_keep={per_keep_bytes:.2f} = {estimated_num_keep}")
        logger.info(f"  Final num_keep={new_num_keep} (max={_max_num_keep})")

        # --- Step 5: 返回调整后的 codebook ---
        if new_num_keep == 0:
            return self._adjust_num_keep_with_value(0) + (None, None, None, None)

        (final_cb, final_indices, final_num_keep) = self._adjust_num_keep_with_value(new_num_keep)

        logger.info(f"=== Budget-First allocation complete ===")
        logger.info(f"  num_keep={new_num_keep}/{_max_num_keep} "
                    f"(est. CB cost: +{new_num_keep * per_keep_bytes / 1024 / 1024:.2f}MB)")

        return (final_cb, final_indices, final_num_keep,
                None, None, None, None)

    def _adjust_num_keep_with_value(self, new_num_keep: int):
        """
        使用指定的new_num_keep值进行调整（用于三分搜索）。

        Args:
            new_num_keep: 指定的保留点数量

        Returns:
            adjusted_cb: 调整后的codebook
            adjusted_indices: 调整后的feature_indices
            adjusted_num_keep: 调整后的保留点数量
        """
        # 检查是否支持动态调整
        is_new_format = getattr(self, '_supports_dynamic_adjustment', False)
        if not is_new_format:
            logger.info(f"Dynamic adjustment not available, using original codebook")
            return (self._features_rest, self._feature_indices, self._num_keep)

        # 确保new_num_keep在有效范围内
        new_num_keep = max(0, min(new_num_keep, self._max_num_keep))

        # 如果不需要调整
        if new_num_keep >= self._max_num_keep:
            return (self._features_rest, self._feature_indices, self._num_keep)

        # 从纯VQ codebook开始
        vq_codebook = self._original_codebook
        vq_cb_size = vq_codebook.shape[0]

        # 从当前codebook中提取kept points的原始SH特征
        current_cb = self._features_rest
        kept_features_full = current_cb[vq_cb_size:vq_cb_size + self._max_num_keep]

        # 选择要保留的kept points（根据importance排序）
        if new_num_keep > 0:
            kept_mask = self._keep_mask
            kept_imp = self.imp[kept_mask]
            _, topk_indices = torch.topk(kept_imp, k=new_num_keep, largest=True)
            selected_kept_features = kept_features_full[topk_indices]
            new_codebook = torch.cat([vq_codebook, selected_kept_features], dim=0)
        else:
            new_codebook = vq_codebook

        # 构建新索引
        new_indices = self._vq_indices_for_all.clone()
        if new_num_keep > 0:
            kept_mask = self._keep_mask
            kept_imp = self.imp[kept_mask]
            _, topk_indices = torch.topk(kept_imp, k=new_num_keep, largest=True)
            kept_indices = torch.where(kept_mask)[0]
            selected_global_indices = kept_indices[topk_indices]
            new_indices[selected_global_indices] = torch.arange(
                vq_cb_size, vq_cb_size + new_num_keep,
                dtype=torch.long, device=vq_codebook.device
            )

        return (new_codebook, new_indices, new_num_keep)

    def _encode_with_params(self, tmp_file: io.BufferedWriter, new_num_keep: int, qbits=None):
        """
        使用指定的new_num_keep和qbits进行编码。

        Args:
            tmp_file: 输出文件
            new_num_keep: 指定的保留点数量
            qbits: 量化位宽（如果为None则使用search_qbits搜索）
        """
        logger.info(f"xyz shape {self.xyz.shape}")
        with tempfile.TemporaryDirectory() as exp_dir:
            os.makedirs(exp_dir, exist_ok=True)
            bin_dir = os.path.join(exp_dir, 'bins')
            os.makedirs(bin_dir, exist_ok=True)
            trans_array = []
            trans_array.append(self.depth)
            trans_array.append(self.n_block)

            with torch.no_grad():
                # Octree: use GPCC encoding (fallback to zlib)
                if is_gpcc_available():
                    try:
                        oct_bitstream = gpcc_encode_octree(self.oct, self.oct_param, self.depth)
                        with open(os.path.join(bin_dir, 'oct.gpcc'), 'wb') as f_oct:
                            f_oct.write(oct_bitstream)
                        logger.info(f"GPCC encoded octree: {len(oct_bitstream)} bytes")
                    except Exception as e:
                        logger.warning(f"GPCC encode failed, falling back to zlib: {e}")
                        np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)
                else:
                    np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)

                # 使用指定的new_num_keep进行调整
                (adjusted_cb_tensor, adjusted_indices_tensor, adjusted_num_keep) = \
                    self._adjust_num_keep_with_value(new_num_keep)

                ntk = adjusted_indices_tensor.detach().contiguous().cpu().int().numpy()
                cb = adjusted_cb_tensor.detach().contiguous().cpu().numpy()

                # 保存 VQ indices (Laplace range coding)
                ntk_bitstream = ntk_encode(ntk, n_block=self.ntk_n_block)
                with open(os.path.join(bin_dir, 'ntk.bin'), 'wb') as f_ntk:
                    f_ntk.write(ntk_bitstream)
                logger.info(f"Laplace encoded NTK: {len(ntk_bitstream)} bytes")

                # 对整个 codebook 进行量化
                if adjusted_num_keep > 0:
                    from .meson_utils import quantize_kept_sh
                    cb_torch = torch.from_numpy(cb).cuda()
                    cb_q_indices, cb_scales, cb_zero_points, cb_split = quantize_kept_sh(
                        cb_torch,
                        n_block=self.cb_n_block,
                        num_bits=self.cb_quant_bits
                    )

                    cb_q_np = cb_q_indices.detach().cpu().numpy()
                    cb_scales_np = cb_scales.detach().cpu().numpy()
                    cb_zps_np = cb_zero_points.detach().cpu().numpy()

                    # Laplace 编码 codebook 量化索引
                    cb_q_bitstream = laplace_encode_blocks(cb_q_np, cb_split, cb_q_np.shape[0])
                    with open(os.path.join(bin_dir, 'cb_q.bin'), 'wb') as f_cb:
                        f_cb.write(cb_q_bitstream)
                    
                    np.savez_compressed(
                        os.path.join(bin_dir, 'cb_meta.npz'),
                        scales=cb_scales_np,
                        zero_points=cb_zps_np,
                        split=np.array(cb_split),
                        num_entries=cb.shape[0],
                        num_keep=adjusted_num_keep,
                        cb_quant_bits=self.cb_quant_bits
                    )
                else:
                    np.savez_compressed(os.path.join(bin_dir , 'um.npz'), umap=cb)

                # 准备RAHT变换的输入
                if self.use_quat:
                    rf = torch.concat([self.original_opacity.detach(), self.original_rotation.detach(), self.features_dc.detach().contiguous().squeeze()], axis=-1)
                else:
                    r = self.original_rotation
                    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                    q = r / norm[:, None]
                    eulers = ToEulerAngles_FT(q)
                    rf = torch.concat([self.original_opacity.detach(), eulers.detach(), self.features_dc.detach().contiguous().squeeze()], axis=-1)

                C = rf[self.reorder]
                if self.raht:
                    iW1 = self.res['iW1']
                    iW2 = self.res['iW2']
                    iLeft_idx = self.res['iLeft_idx']
                    iRight_idx = self.res['iRight_idx']

                    for d in range(self.depth * 3):
                        w1 = iW1[d]
                        w2 = iW2[d]
                        left_idx = iLeft_idx[d]
                        right_idx = iRight_idx[d]
                        C[left_idx], C[right_idx] = transform_batched_torch(w1, w2, C[left_idx], C[right_idx])
                else:
                    logger.info("RAHT ablation: skipping RAHT forward transform in _encode_with_params")

                cf = C[0].cpu().numpy()

                channels = C.shape[-1]
                split = split_length(C.shape[0] - 1, self.n_block)

                # 如果没有提供qbits，则进行搜索
                if qbits is None and self.size_limit_mb > 0:
                    logger.info(f"Searching qbits with limit {self.size_limit_mb} MB, num_keep={adjusted_num_keep}")
                    qbits, _obj_value = search_qbits(
                        n_round=5,
                        depth=self.depth,
                        n_block=self.n_block,
                        oct=self.oct,
                        oct_param=self.oct_param,
                        fdc=self.features_dc,
                        opa=self.original_opacity,
                        scales=self.original_scales,
                        r=self.original_rotation,
                        ntk=ntk,
                        cb=cb,
                        low_bit=1,
                        high_bit=self.num_bits,
                        size_limit_mb=self.size_limit_mb,
                        search_rf=True,
                        search_scale=True,
                        equal_bit_val=self.num_bits,
                        use_quat=self.use_quat,
                        num_keep=adjusted_num_keep,
                        cb_quant_bits=self.cb_quant_bits,
                        use_raht=self.raht,
                        channel_importance_weight=self.channel_importance_weight,
                        percentile_quant=self.percentile_quant,
                        auto_entropy_model=self.auto_entropy_model
                    )
                    rf_channels = 8 if self.use_quat else 7
                    qbits_rf = qbits[:rf_channels]
                    qbits_scale = qbits[rf_channels:]
                elif qbits is not None:
                    rf_channels = 8 if self.use_quat else 7
                    qbits_rf = qbits[:rf_channels]
                    qbits_scale = qbits[rf_channels:]
                else:
                    qbits_rf = None
                    qbits_scale = None

                # 量化 orgb
                splits_tensor = torch.tensor(split, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels, 1)
                if qbits_rf is not None:
                    power_qbits_tensor = (2 ** torch.tensor(qbits_rf, device='cuda') - 1).int()
                    max_bit_rf = int(qbits_rf.max()) if qbits_rf.size > 0 else self.num_bits
                    q_dtype = get_dtype_for_bits(max_bit_rf)
                else:
                    power_qbits_tensor = torch.full((channels, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)
                    q_dtype = get_dtype_for_bits(self.num_bits)

                q_indices, scales, zps, _, _ = pure_quant_wo_minmax(C[1:], splits_tensor, power_qbits_tensor)
                qci_2d = q_indices.cpu().numpy()  # [channels, N]
                trans_tensor = torch.stack([scales, zps], dim=-1).flatten()
                trans_array.extend(trans_tensor.cpu().tolist())
                # Laplace 编码 orgb
                orgb_bitstream = laplace_encode_blocks(qci_2d, split, channels, auto_model=self.auto_entropy_model)
                with open(os.path.join(bin_dir, 'orgb.bin'), 'wb') as f_orgb:
                    f_orgb.write(orgb_bitstream)
                np.savez_compressed(os.path.join(bin_dir, 'orgb_dc.npz'), f=cf)

                # 量化 scaling
                scaling = self.original_scales.detach()
                channels_scale = scaling.shape[-1]
                split_scale = split_length(scaling.shape[0], self.n_block)
                splits_tensor_scale = torch.tensor(split_scale, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels_scale, 1)
                if qbits_scale is not None:
                    power_qbits_scale = (2 ** torch.tensor(qbits_scale, device='cuda') - 1).int()
                    max_bit_scale = int(qbits_scale.max()) if qbits_scale.size > 0 else self.num_bits
                    q_dtype_scale = get_dtype_for_bits(max_bit_scale)
                else:
                    power_qbits_scale = torch.full((channels_scale, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)
                    q_dtype_scale = get_dtype_for_bits(self.num_bits)

                q_indices_s, scales_s, zps_s, _, _ = pure_quant_wo_minmax(scaling, splits_tensor_scale, power_qbits_scale)
                scaling_q_2d = q_indices_s.cpu().numpy()  # [channels_scale, N]
                trans_tensor_s = torch.stack([scales_s, zps_s], dim=-1).flatten()
                trans_array.extend(trans_tensor_s.cpu().tolist())
                # Laplace 编码 ct
                ct_bitstream = laplace_encode_blocks(scaling_q_2d, split_scale, channels_scale, auto_model=self.auto_entropy_model)
                with open(os.path.join(bin_dir, 'ct.bin'), 'wb') as f_ct:
                    f_ct.write(ct_bitstream)

                trans_array = np.array(trans_array)
                np.savez_compressed(os.path.join(bin_dir, 't.npz'), t=trans_array)

                # 保存 ILP 搜索的 qbits 分配结果（用于可视化/分析）
                # 不能落在 exp_dir，因为它是 TemporaryDirectory，函数结束会被删除。
                if qbits is not None:
                    qbits_output_dir = exp_dir
                    tmp_output_path = getattr(tmp_file, 'name', None)
                    if isinstance(tmp_output_path, str) and tmp_output_path:
                        qbits_output_dir = os.path.dirname(os.path.abspath(tmp_output_path))
                    qbits_path = os.path.join(qbits_output_dir, 'qbits.npz')
                    np.savez_compressed(qbits_path, qbits=qbits)
                    logger.info(f"Saved qbits allocation to standalone file: {qbits_path}, shape={qbits.shape}, "
                               f"min={qbits.min()}, max={qbits.max()}, mean={qbits.mean():.2f}")

                bin_zip_name = bin_dir.split('/')[-1]
                bin_zip_path = os.path.join(exp_dir, f'{bin_zip_name}.zip')
                os.system(f'zip -j {bin_zip_path} {bin_dir}/*')

                zip_file_size = os.path.getsize(bin_zip_path)
                logger.info(f'Encoded size: {zip_file_size / 1024 / 1024:.2f} MB')

                with open(bin_zip_path, "rb") as f_in:
                    while True:
                        chunk = f_in.read(8192)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                tmp_file.flush()
                return zip_file_size

    def _get_objective_for_num_keep(self, new_num_keep, prev_qbits=None):
        """
        使用指定的 new_num_keep 调用 search_qbits，返回 objective value 和 qbits。

        Args:
            new_num_keep: 指定的保留点数量
            prev_qbits: 上一次搜索的 qbits 结果，用作 warm start

        Returns:
            obj_value: search_qbits 的 objective value
            qbits: 搜索得到的量化位宽
        """
        # 调整 num_keep
        (adjusted_cb_tensor, adjusted_indices_tensor, adjusted_num_keep) = \
            self._adjust_num_keep_with_value(new_num_keep)

        ntk = adjusted_indices_tensor.detach().contiguous().cpu().int().numpy()
        cb = adjusted_cb_tensor.detach().contiguous().cpu().numpy()

        # 调用 search_qbits
        qbits, obj_value = search_qbits(
            n_round=5,
            depth=self.depth,
            n_block=self.n_block,
            oct=self.oct,
            oct_param=self.oct_param,
            fdc=self.features_dc,
            opa=self.original_opacity,
            scales=self.original_scales,
            r=self.original_rotation,
            ntk=ntk,
            cb=cb,
            low_bit=1,
            high_bit=self.num_bits,
            size_limit_mb=self.size_limit_mb,
            search_rf=True,
            search_scale=True,
            equal_bit_val=self.num_bits,
            use_quat=self.use_quat,
            num_keep=adjusted_num_keep,
            cb_quant_bits=self.cb_quant_bits,
            init_qbits=prev_qbits,
            use_raht=self.raht,
            channel_importance_weight=self.channel_importance_weight,
            percentile_quant=self.percentile_quant,
            auto_entropy_model=self.auto_entropy_model
        )

        # 检测无效解：qbits 全 0（或全 1，即所有 block 选择最低 bit）
        # 这表示 size_limit 太小，无法容纳 num_keep 这么多点的正常量化
        if qbits is not None and np.all(qbits <= 1):
            logger.warning(f"Invalid qbits detected (all <= 1) for num_keep={new_num_keep}, "
                          f"treating as infeasible. qbits sample: {qbits[0, :10]}")
            obj_value = float('inf')
        
        return obj_value, qbits

    @torch.no_grad
    def _binary_search_num_keep(self):
        """
        使用二分法搜索最优的 new_num_keep 值。
        
        判断标准：先计算 num_keep=0 时的基线 obj_base，然后阈值 = obj_base * obj_threshold_ratio。
        这样阈值自动适配不同场景大小，无需手动缩放。
        - obj_value > threshold → 保留点太多，搜索左半区间
        - obj_value <= threshold → 可接受，搜索右半区间（保留点越多越好）
        - 区间宽度 < binary_search_interval 时退出，以左端点作为最终值

        Returns:
            optimal_new_num_keep: 最优的 new_num_keep 值
            best_qbits: 对应的 qbits 结果
        """
        logger.info("=== Starting Binary Search for new_num_keep ===")
        logger.info(f"Search range: [0, {self._max_num_keep}]")
        logger.info(f"Stop interval: {self.binary_search_interval}")
        logger.info(f"Threshold ratio: {self.obj_threshold_ratio}")

        # P0: 估算 codebook 容量上界，防止 num_keep 过大导致 codebook 超过 size_limit（考虑 cb_quant_bits）
        vq_cb_size = self._original_codebook.shape[0]
        cb_dim = self._original_codebook.shape[1]
        bytes_per_element = self.cb_quant_bits / 8.0  # 每个量化元素占用的字节数
        per_keep_bytes = cb_dim * bytes_per_element
        vq_cb_bytes = vq_cb_size * cb_dim * bytes_per_element
        max_cb_budget_bytes = self.size_limit_mb * 1024 * 1024 * 0.8
        max_keep_by_budget = max(0, int((max_cb_budget_bytes - vq_cb_bytes) / per_keep_bytes)) if per_keep_bytes > 0 else self._max_num_keep
        logger.info(f"CB budget estimation: cb_quant_bits={self.cb_quant_bits}, bytes_per_element={bytes_per_element:.2f}, "
                    f"vq_cb_bytes={vq_cb_bytes/1024/1024:.2f}MB, per_keep_bytes={per_keep_bytes:.1f}, "
                    f"max_keep_by_budget={max_keep_by_budget}")
        effective_max_keep = min(self._max_num_keep, max_keep_by_budget)
        if effective_max_keep < self._max_num_keep:
            logger.warning(f"Binary search upper bound capped: {self._max_num_keep} -> {effective_max_keep} "
                          f"(size_limit={self.size_limit_mb}MB)")

        # Step 1: 计算基线 objective value（num_keep=0，即不保留任何点）
        logger.info("Computing baseline objective value with num_keep=0...")
        obj_base, base_qbits = self._get_objective_for_num_keep(0, prev_qbits=None)
        threshold = obj_base * self.obj_threshold_ratio
        logger.info(f"Baseline obj_base={obj_base:.6f}, threshold={threshold:.6f} (ratio={self.obj_threshold_ratio})")

        left = 0
        right = effective_max_keep
        iteration = 0
        prev_qbits = base_qbits
        best_qbits = base_qbits  # num_keep=0 肯定可行

        while (right - left) >= self.binary_search_interval:
            iteration += 1
            mid = (left + right) // 2

            logger.info(f"\n--- Binary Search Iteration {iteration} ---")
            logger.info(f"Current interval: [{left}, {right}], width: {right - left}, mid: {mid}")

            # 使用 mid 值调用 search_qbits
            obj_value, qbits = self._get_objective_for_num_keep(mid, prev_qbits)

            logger.info(f"  num_keep={mid}: obj_value={obj_value:.6f}, "
                       f"threshold={threshold:.6f} (base={obj_base:.6f} * ratio={self.obj_threshold_ratio})")

            if obj_value > threshold:
                # 保留点太多，量化空间不足，objective value 过大
                right = mid
                logger.info(f"  obj_value > threshold, searching in [{left}, {mid}]")
            else:
                # objective value 可接受，尝试保留更多点
                left = mid
                best_qbits = qbits  # 记录可行的 qbits
                logger.info(f"  obj_value <= threshold, searching in [{mid}, {right}]")

            # 复用当前 qbits 作为下次 warm start
            prev_qbits = qbits

            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"\n=== Binary Search Complete ===")
        logger.info(f"Optimal new_num_keep: {left}")
        logger.info(f"Final interval: [{left}, {right}]")

        return left, best_qbits

    def _golden_section_search_num_keep(self, init_model_fn, mp, scene, ppl, opt):
        """
        使用黄金分割搜索最优的new_num_keep值（找PSNR最大值，单峰假设）。

        相比三分搜索，黄金分割搜索利用黄金比例的性质，使得每次迭代只需新评估1个点
        （上一轮的探测点可以复用），总评估次数约为三分搜索的一半。

        Args:
            init_model_fn: 用于创建新模型实例的函数
            mp: ModelParams
            scene: 场景对象，用于评估
            ppl: PipelineParams，用于评估
            opt: OptimizationParams，用于评估

        Returns:
            optimal_new_num_keep: 最优的new_num_keep值
        """
        import math

        phi = (1 + math.sqrt(5)) / 2  # ≈ 1.618
        resphi = 2 - phi               # ≈ 0.382

        logger.info(f"=== Starting Golden Section Search for new_num_keep ===")
        logger.info(f"Original max_num_keep: {self._max_num_keep}")
        logger.info(f"Stop interval: {self.golden_search_interval}")

        # 基于 codebook budget 估算上界（考虑 cb_quant_bits）
        vq_cb_size = self._original_codebook.shape[0]
        cb_dim = self._original_codebook.shape[1]
        bytes_per_element = self.cb_quant_bits / 8.0  # 每个量化元素占用的字节数
        per_keep_bytes = cb_dim * bytes_per_element
        vq_cb_bytes = vq_cb_size * cb_dim * bytes_per_element
        max_cb_budget_bytes = self.size_limit_mb * 1024 * 1024 * 0.8
        max_keep_by_budget = max(0, int((max_cb_budget_bytes - vq_cb_bytes) / per_keep_bytes)) if per_keep_bytes > 0 else self._max_num_keep
        logger.info(f"CB budget estimation: cb_quant_bits={self.cb_quant_bits}, bytes_per_element={bytes_per_element:.2f}, "
                    f"vq_cb_bytes={vq_cb_bytes/1024/1024:.2f}MB, per_keep_bytes={per_keep_bytes:.1f}, "
                    f"max_keep_by_budget={max_keep_by_budget}")
        effective_max_keep = min(self._max_num_keep, max_keep_by_budget)

        effective_min_keep = 0

        logger.info(f"Search range: [{effective_min_keep}, {effective_max_keep}]")
        if effective_max_keep < self._max_num_keep:
            logger.warning(f"Golden section search upper bound capped: {self._max_num_keep} -> {effective_max_keep} "
                          f"(size_limit={self.size_limit_mb}MB)")

        if effective_max_keep <= effective_min_keep:
            logger.warning(f"No room for search (lower={effective_min_keep}, upper={effective_max_keep}), returning {effective_min_keep}")
            return effective_min_keep

        left = effective_min_keep
        right = effective_max_keep

        # 初始两个探测点：x1 < x2
        x1 = left + int(resphi * (right - left))
        x2 = right - int(resphi * (right - left))

        # 处理整数离散化导致 x1 == x2 的边界情况
        if x1 == x2:
            if x2 < right:
                x2 = x1 + 1
            else:
                x1 = x2 - 1

        logger.info(f"Initial probe points: x1={x1}, x2={x2}")

        # 初始化需要评估两个点
        f1 = self._evaluate_with_num_keep(x1, init_model_fn, mp, scene, ppl, opt)
        logger.info(f"  x1={x1}: PSNR={f1:.4f}")
        f2 = self._evaluate_with_num_keep(x2, init_model_fn, mp, scene, ppl, opt)
        logger.info(f"  x2={x2}: PSNR={f2:.4f}")

        best_psnr = max(f1, f2)
        best_num_keep = x1 if f1 >= f2 else x2
        iteration = 0

        while (right - left) >= self.golden_search_interval:
            iteration += 1
            logger.info(f"\n--- Golden Section Iteration {iteration} ---")
            logger.info(f"Current interval: [{left}, {right}], width: {right - left}")

            if f1 < f2:
                # 极大值在 [x1, right]，丢弃左侧
                left = x1
                x1 = x2       # 复用 x2 → 新的 x1
                f1 = f2
                x2 = right - int(resphi * (right - left))
                # 处理整数离散化
                if x2 <= x1:
                    x2 = x1 + 1 if x1 + 1 <= right else x1
                if x2 == x1:
                    break
                f2 = self._evaluate_with_num_keep(x2, init_model_fn, mp, scene, ppl, opt)
                logger.info(f"  Reuse x1={x1} (PSNR={f1:.4f}), new x2={x2}: PSNR={f2:.4f}")
            else:
                # 极大值在 [left, x2]，丢弃右侧
                right = x2
                x2 = x1       # 复用 x1 → 新的 x2
                f2 = f1
                x1 = left + int(resphi * (right - left))
                # 处理整数离散化
                if x1 >= x2:
                    x1 = x2 - 1 if x2 - 1 >= left else x2
                if x1 == x2:
                    break
                f1 = self._evaluate_with_num_keep(x1, init_model_fn, mp, scene, ppl, opt)
                logger.info(f"  New x1={x1}: PSNR={f1:.4f}, reuse x2={x2} (PSNR={f2:.4f})")

            # 更新最优值
            if f1 > best_psnr:
                best_psnr = f1
                best_num_keep = x1
            if f2 > best_psnr:
                best_psnr = f2
                best_num_keep = x2

            # 清理GPU缓存
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"\n=== Golden Section Search Complete ===")
        logger.info(f"Optimal new_num_keep: {best_num_keep}")
        logger.info(f"Best PSNR: {best_psnr:.4f}")
        logger.info(f"Final interval: [{left}, {right}]")

        return best_num_keep

    def _evaluate_with_num_keep(self, new_num_keep, init_model_fn, mp, scene, ppl, opt):
        """
        使用指定的new_num_keep值进行编码、解码和评估。

        Args:
            new_num_keep: 指定的保留点数量
            init_model_fn: 用于创建新模型实例的函数
            mp: ModelParams
            scene: 场景对象
            ppl: PipelineParams
            opt: OptimizationParams

        Returns:
            psnr: 评估得到的PSNR值，失败时返回 -1.0
        """
        from splatwizard.pipeline.evaluation import evaluate

        try:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as tmp_file:
                # 使用指定的new_num_keep进行编码
                logger.info(f"  Evaluating with new_num_keep={new_num_keep}")
                self._encode_with_params(tmp_file, new_num_keep)
                encoded_size = tmp_file.tell()

                # 创建新的模型实例并解码
                tmp_file.seek(0)

                # 使用与eval_model.py相同的方式创建新实例
                class DummyTrainContext:
                    def __init__(self):
                        self.model = 'mesongs_plus'

                new_model = init_model_fn(DummyTrainContext().model, mp)
                new_model.spatial_lr_scale = scene.cameras_extent

                with torch.no_grad():
                    new_model.decode(tmp_file)

                # 评估模型
                eval_pack = evaluate(new_model, ppl, scene, None)
                psnr = eval_pack.psnr_val

                logger.info(f"  Evaluation complete: PSNR={psnr:.4f}, Size={encoded_size/1024/1024:.2f}MB")

                del new_model
                torch.cuda.empty_cache()

                return psnr
        except Exception as e:
            logger.error(f"  Evaluation FAILED for new_num_keep={new_num_keep}: {e}")
            torch.cuda.empty_cache()
            return -1.0

    @torch.no_grad
    def encode(self, tmp_file: io.BufferedWriter, scene=None, ppl=None, opt=None, mp=None, init_model_fn=None, stage_timings: dict = None):
        """
        Encode the Gaussian model to a compressed bitstream.
        
        Args:
            stage_timings: If provided (dict), each sub-stage's wall-clock time (seconds)
                          will be recorded into this dict. Keys include:
                          'binary_search', 'octree_gpcc', 'ntk_laplace', 'codebook_quant',
                          'quat_to_euler', 'raht', 'ilp_search', 'segquant_rf', 'laplace_orgb',
                          'quant_scales', 'laplace_scales', 'zip_pack'
        """
        import time as _time

        def _tick():
            torch.cuda.synchronize()
            return _time.time()

        def _record(name, t0):
            """Record elapsed time since t0 into stage_timings[name]."""
            if stage_timings is not None:
                torch.cuda.synchronize()
                stage_timings[name] = _time.time() - t0

        # path = tmp_file.name
        logger.info(f"xyz shape {self.xyz.shape}")

        # === Phase 0: Eval-time re-pruning（手动指定 pruning_rate）===
        if self.pruning_rate > 0.0:
            logger.info(f"=== Eval-time re-pruning with pruning_rate={self.pruning_rate} ===")
            self.re_prune_and_rebuild_octree(self.pruning_rate)
            logger.info(f"After re-pruning: xyz shape {self.xyz.shape}")

        # 搜索最优 new_num_keep
        optimal_new_num_keep = None
        binary_search_qbits = None  # 二分搜索得到的 qbits，可以复用到最终 encode
        
        # 如果 CLI 直接指定了 num_keep，优先使用
        if self.num_keep >= 0:
            optimal_new_num_keep = self.num_keep
            logger.info(f"Using CLI-specified num_keep={self.num_keep}")
        # 二分搜索优先级高于黄金分割搜索
        elif self.enable_binary_search and self.size_limit_mb > 0:
            logger.info("Binary search enabled, searching for optimal new_num_keep...")
            optimal_new_num_keep, binary_search_qbits = self._binary_search_num_keep()
            logger.info(f"Binary search found optimal new_num_keep={optimal_new_num_keep}")
        elif self.enable_golden_search and scene is not None and ppl is not None and opt is not None and mp is not None and init_model_fn is not None:
            if self.size_limit_mb > 0:
                logger.info("Golden section search enabled, searching for optimal new_num_keep...")
                optimal_new_num_keep = self._golden_section_search_num_keep(
                    init_model_fn, mp, scene, ppl, opt
                )
                logger.info(f"Golden section search found optimal new_num_keep={optimal_new_num_keep}")
            else:
                logger.info("Golden section search enabled but size_limit_mb <= 0, skipping")

        with tempfile.TemporaryDirectory() as exp_dir:
            os.makedirs(exp_dir, exist_ok=True)
            bin_dir = os.path.join(exp_dir, 'bins')
            os.makedirs(bin_dir, exist_ok=True)
            trans_array = []
            trans_array.append(self.depth)
            trans_array.append(self.n_block)

            scale_offset = 7

            with torch.no_grad():
                # Octree: use GPCC encoding (fallback to zlib)
                _t0 = _tick()
                if is_gpcc_available():
                    try:
                        oct_bitstream = gpcc_encode_octree(self.oct, self.oct_param, self.depth)
                        with open(os.path.join(bin_dir, 'oct.gpcc'), 'wb') as f_oct:
                            f_oct.write(oct_bitstream)
                        logger.info(f"GPCC encoded octree: {len(oct_bitstream)} bytes")
                    except Exception as e:
                        logger.warning(f"GPCC encode failed, falling back to zlib: {e}")
                        np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)
                else:
                    np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)
                _record('octree_gpcc', _t0)

                # 动态调整保留点数量
                # 如果搜索（二分/三分）找到了最优值，使用它；否则使用原有逻辑
                if optimal_new_num_keep is not None:
                    logger.info(f"Using optimal new_num_keep from search: {optimal_new_num_keep}")
                    (adjusted_cb_tensor, adjusted_indices_tensor, adjusted_num_keep) = \
                        self._adjust_num_keep_with_value(optimal_new_num_keep)
                else:
                    (adjusted_cb_tensor, adjusted_indices_tensor, adjusted_num_keep,
                     adjusted_q_indices, adjusted_scales, adjusted_zero_points, adjusted_split) = \
                        self._adjust_num_keep_for_size_limit()

                ntk = adjusted_indices_tensor.detach().contiguous().cpu().int().numpy()
                cb = adjusted_cb_tensor.detach().contiguous().cpu().numpy()

                # 保存 VQ indices (Laplace range coding)
                _t0 = _tick()
                ntk_bitstream = ntk_encode(ntk, n_block=self.ntk_n_block)
                with open(os.path.join(bin_dir, 'ntk.bin'), 'wb') as f_ntk:
                    f_ntk.write(ntk_bitstream)
                logger.info(f"Laplace encoded NTK: {len(ntk_bitstream)} bytes (n_block={self.ntk_n_block})")
                _record('ntk_laplace', _t0)

                # 对整个 codebook 进行量化（包括 kept features 和 VQ codebook）
                _t0 = _tick()
                if adjusted_num_keep > 0:
                    logger.info(f"Quantizing entire codebook ({cb.shape[0]} entries including {adjusted_num_keep} kept points)")

                    # 对整个 cb 进行量化（不管是否预先计算了量化参数）
                    from .meson_utils import quantize_kept_sh
                    cb_torch = torch.from_numpy(cb).cuda()
                    cb_q_indices, cb_scales, cb_zero_points, cb_split = quantize_kept_sh(
                        cb_torch,
                        n_block=self.cb_n_block,
                        num_bits=self.cb_quant_bits
                    )

                    # 保存量化后的 codebook（使用 Laplace 范围编码）
                    cb_q_np = cb_q_indices.detach().cpu().numpy()
                    cb_scales_np = cb_scales.detach().cpu().numpy()
                    cb_zps_np = cb_zero_points.detach().cpu().numpy()

                    # Laplace 编码 codebook 量化索引
                    cb_q_bitstream = laplace_encode_blocks(cb_q_np, cb_split, cb_q_np.shape[0])
                    with open(os.path.join(bin_dir, 'cb_q.bin'), 'wb') as f_cb:
                        f_cb.write(cb_q_bitstream)
                    
                    # 元信息仍用 npz 保存（scales, zero_points, split 等）
                    np.savez_compressed(
                        os.path.join(bin_dir, 'cb_meta.npz'),
                        scales=cb_scales_np,
                        zero_points=cb_zps_np,
                        split=np.array(cb_split),
                        num_entries=cb.shape[0],
                        num_keep=adjusted_num_keep,
                        cb_quant_bits=self.cb_quant_bits
                    )

                    logger.info(f"  Codebook quantized: shape={cb_q_np.shape}, dtype={cb_q_np.dtype}")
                    logger.info(f"  Codebook Laplace encoded: {len(cb_q_bitstream)} bytes")
                    logger.info(f"  Codebook scales/zps: shape={cb_scales_np.shape}")

                    # 不再保存 um.npz（float32 codebook）
                else:
                    # 原始模式：保存完整 codebook（无TopK量化）
                    np.savez_compressed(os.path.join(bin_dir , 'um.npz'), umap=cb)

                # 计算 SH 压缩数据大小
                ntk_size = os.path.getsize(os.path.join(bin_dir, 'ntk.bin'))

                # 如果使用了 TopK 量化（整个 codebook 量化）
                if adjusted_num_keep > 0:
                    cb_q_size = os.path.getsize(os.path.join(bin_dir, 'cb_q.bin'))
                    cb_meta_size = os.path.getsize(os.path.join(bin_dir, 'cb_meta.npz'))
                    cb_total_size = cb_q_size + cb_meta_size
                    sh_total_size = ntk_size + cb_total_size
                else:
                    cb_size = os.path.getsize(os.path.join(bin_dir, 'um.npz'))
                    sh_total_size = ntk_size + cb_size

                # 分析数据结构
                num_points = ntk.shape[0]
                sh_dim = cb.shape[1] if cb.ndim > 1 else 0
                original_sh_size_uncompressed = num_points * sh_dim * 4 if sh_dim > 0 else 0

                logger.info(f"=== SH (Spherical Harmonics) Compression Stats ===")
                logger.info(f"Total points: {num_points}")

                if adjusted_num_keep > 0:
                    # TopK 量化模式（整个 codebook 量化）
                    num_cb_entries = cb.shape[0] if cb.ndim > 1 else 0
                    logger.info(f"Mode: TopK with Full Codebook Quantization (Laplace Range Coded)")
                    logger.info(f"  - Kept points: {adjusted_num_keep}")
                    logger.info(f"  - Total codebook entries: {num_cb_entries}")
                    logger.info(f"  - SH dimension: {sh_dim}")

                    logger.info(f"Compressed sizes:")
                    logger.info(f"  - Indices (ntk): {ntk_size} bytes ({ntk_size/1024:.2f} KB, {ntk_size/1024/1024:.2f} MB)")
                    logger.info(f"      shape={ntk.shape}, dtype={ntk.dtype}")

                    logger.info(f"  - Codebook (Laplace coded): {cb_q_size} bytes ({cb_q_size/1024:.2f} KB, {cb_q_size/1024/1024:.2f} MB)")
                    logger.info(f"  - Codebook meta: {cb_meta_size} bytes ({cb_meta_size/1024:.2f} KB)")
                    logger.info(f"      q_indices shape={cb_q_np.shape}, dtype={cb_q_np.dtype}")
                    logger.info(f"      scales/zps shape={cb_scales_np.shape}")
                    cb_original = num_cb_entries * sh_dim * 4
                    logger.info(f"      Original (float32): {cb_original / 1024 / 1024:.2f} MB")
                    logger.info(f"      Compression: {cb_original / cb_total_size:.2f}x")
                else:
                    # 原始 VQ 模式
                    num_codebook_entries = cb.shape[0] if cb.ndim > 1 else 0
                    logger.info(f"Mode: Standard VQ")
                    logger.info(f"Codebook entries: {num_codebook_entries} (SH dim={sh_dim})")

                    logger.info(f"Compressed sizes:")
                    logger.info(f"  - Indices (ntk): {ntk_size} bytes ({ntk_size/1024:.2f} KB, {ntk_size/1024/1024:.2f} MB)")
                    logger.info(f"  - Codebook (cb): {cb_size} bytes ({cb_size/1024:.2f} KB, {cb_size/1024/1024:.2f} MB)")

                logger.info(f"  - SH Total Compressed: {sh_total_size} bytes ({sh_total_size/1024:.2f} KB, {sh_total_size/1024/1024:.2f} MB)")
                if original_sh_size_uncompressed > 0:
                    logger.info(f"  - SH Original (uncompressed): {original_sh_size_uncompressed} bytes ({original_sh_size_uncompressed/1024/1024:.2f} MB)")
                    logger.info(f"  - SH Compression ratio: {original_sh_size_uncompressed / sh_total_size:.2f}x")
                logger.info(f"=" * 50)
                _record('codebook_quant', _t0)
                
                _t0 = _tick()
                if self.use_quat:
                    rf = torch.concat([self.original_opacity.detach(), self.original_rotation.detach(), self.features_dc.detach().contiguous().squeeze()], axis=-1)
                else:
                    r = self.original_rotation
                    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                    q = r / norm[:, None]
                    eulers = ToEulerAngles_FT(q)
                    
                    rf = torch.concat([self.original_opacity.detach(), eulers.detach(), self.features_dc.detach().contiguous().squeeze()], axis=-1)
                _record('quat_to_euler', _t0)
                    
                # '''ckpt'''
                # rf_cpu = rf.cpu().numpy()
                # np.save('duipai/rf_cpu.npy', rf_cpu)
                # ''''''
                
                C = rf[self.reorder]
                _t0 = _tick()
                if self.raht:
                    iW1 = self.res['iW1']
                    iW2 = self.res['iW2']
                    iLeft_idx = self.res['iLeft_idx']
                    iRight_idx = self.res['iRight_idx']

                    for d in range(self.depth * 3):
                        w1 = iW1[d]
                        w2 = iW2[d]
                        left_idx = iLeft_idx[d]
                        right_idx = iRight_idx[d]
                        C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                            w2, 
                                                            C[left_idx], 
                                                            C[right_idx])
                else:
                    logger.info("RAHT ablation: skipping RAHT forward transform in encode")
                _record('raht', _t0)

                cf = C[0].cpu().numpy()

                # SegQuant Encode C
                qa_cnt = 0
                lc1 = C.shape[0] - 1
                channels = C.shape[-1]
                qci = [] 
                split = split_length(lc1, self.n_block)
                
                qbits_rf = None
                qbits_scale = None
                
                # 如果二分搜索已经得到了 qbits，直接复用，避免重复搜索
                _t0 = _tick()
                if binary_search_qbits is not None:
                    logger.info("Reusing qbits from binary search (skipping search_qbits)")
                    qbits = binary_search_qbits
                    rf_channels = 8 if self.use_quat else 7
                    qbits_rf = qbits[:rf_channels]
                    qbits_scale = qbits[rf_channels:]
                elif self.size_limit_mb > 0:
                    logger.info(f"Searching qbits with limit {self.size_limit_mb} MB")
                    qbits, _obj_value = search_qbits(
                        n_round=5,
                        depth=self.depth,
                        n_block=self.n_block,
                        oct=self.oct,
                        oct_param=self.oct_param,
                        fdc=self.features_dc,
                        opa=self.original_opacity,
                        scales=self.original_scales,
                        r=self.original_rotation,
                        ntk=ntk,
                        cb=cb,
                        low_bit=1,
                        high_bit=self.num_bits,
                        size_limit_mb=self.size_limit_mb,
                        search_rf=True,
                        search_scale=True,
                        equal_bit_val=self.num_bits,
                        use_quat=self.use_quat,
                        num_keep=adjusted_num_keep,  # 使用调整后的num_keep
                        cb_quant_bits=self.cb_quant_bits,
                        use_raht=self.raht,
                        channel_importance_weight=self.channel_importance_weight,
                        percentile_quant=self.percentile_quant,
                        auto_entropy_model=self.auto_entropy_model
                    )
                    #对于xyz进行八分树编码
                    #rf是Region adaptive hierarchical transform.变换后的特征(DC,AC信号),对于Opacity, rgb, Euler角(旋转四元数), Scales进行raht编码
                    rf_channels = 8 if self.use_quat else 7
                    qbits_rf = qbits[:rf_channels]
                    qbits_scale = qbits[rf_channels:]
                _record('ilp_search', _t0)

                logger.info(f"encoding orgb.npz with seg-quant")
                splits_tensor = torch.tensor(split, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels, 1)
                if qbits_rf is not None:
                     power_qbits_tensor = (2 ** torch.tensor(qbits_rf, device='cuda') - 1).int()
                     # Use the maximum bit width from qbits_rf for dtype selection
                     max_bit_rf = int(qbits_rf.max()) if qbits_rf.size > 0 else self.num_bits
                     q_dtype = get_dtype_for_bits(max_bit_rf)
                else:
                     power_qbits_tensor = torch.full((channels, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)
                     q_dtype = get_dtype_for_bits(self.num_bits)
                
                _t0 = _tick()
                q_indices, scales, zps, _, _ = pure_quant_wo_minmax(C[1:], splits_tensor, power_qbits_tensor)
                _record('segquant_rf', _t0)
                
                # q_indices shape: [channels, N] from pure_quant_wo_minmax
                qci_2d = q_indices.cpu().numpy()  # [channels, N]
                
                trans_tensor = torch.stack([scales, zps], dim=-1).flatten()
                trans_array.extend(trans_tensor.cpu().tolist())
                
                qa_cnt += channels * self.n_block

                # orgb: Laplace 编码 RAHT AC 系数量化索引 + 保存 DC 系数
                _t0 = _tick()
                orgb_bitstream = laplace_encode_blocks(qci_2d, split, channels, auto_model=self.auto_entropy_model)
                with open(os.path.join(bin_dir, 'orgb.bin'), 'wb') as f_orgb:
                    f_orgb.write(orgb_bitstream)
                np.savez_compressed(os.path.join(bin_dir, 'orgb_dc.npz'), f=cf)
                _record('laplace_orgb', _t0)
                logger.info(f"  orgb Laplace encoded: {len(orgb_bitstream)} bytes (was zlib)")
                
                scaling = self.original_scales.detach()
                lc1 = scaling.shape[0]
                channels_scale = scaling.shape[-1]
                scaling_q = []
                split_scale = split_length(lc1, self.n_block)
                
                splits_tensor_scale = torch.tensor(split_scale, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(channels_scale, 1)
                if qbits_scale is not None:
                    power_qbits_scale = (2 ** torch.tensor(qbits_scale, device='cuda') - 1).int()
                    # Use maximum bit width from qbits_scale for dtype selection
                    max_bit_scale = int(qbits_scale.max()) if qbits_scale.size > 0 else self.num_bits
                    q_dtype_scale = get_dtype_for_bits(max_bit_scale)
                else:
                    power_qbits_scale = torch.full((channels_scale, self.n_block), 2**self.num_bits - 1, device='cuda', dtype=torch.int32)
                    q_dtype_scale = get_dtype_for_bits(self.num_bits)
                
                #开始量化
                _t0 = _tick()
                q_indices_s, scales_s, zps_s, _, _ = pure_quant_wo_minmax(scaling, splits_tensor_scale, power_qbits_scale)
                _record('quant_scales', _t0)
                
                # q_indices_s shape: [channels_scale, N]
                scaling_q_2d = q_indices_s.cpu().numpy()  # [channels_scale, N]
                
                trans_tensor_s = torch.stack([scales_s, zps_s], dim=-1).flatten()
                trans_array.extend(trans_tensor_s.cpu().tolist())
                
                qa_cnt += channels_scale * self.n_block
                # ct: Laplace 编码 scaling 量化索引
                _t0 = _tick()
                ct_bitstream = laplace_encode_blocks(scaling_q_2d, split_scale, channels_scale, auto_model=self.auto_entropy_model)
                with open(os.path.join(bin_dir, 'ct.bin'), 'wb') as f_ct:
                    f_ct.write(ct_bitstream)
                _record('laplace_scales', _t0)
                logger.info(f"  ct Laplace encoded: {len(ct_bitstream)} bytes (was zlib)")
                
                trans_array = np.array(trans_array)
                np.savez_compressed(os.path.join(bin_dir, 't.npz'), t=trans_array)

                # 保存 ILP 搜索的 qbits 分配结果（用于可视化/分析）
                # 不能落在 exp_dir，因为它是 TemporaryDirectory，函数结束会被删除。
                if qbits is not None:
                    qbits_output_dir = exp_dir
                    tmp_output_path = getattr(tmp_file, 'name', None)
                    if isinstance(tmp_output_path, str) and tmp_output_path:
                        qbits_output_dir = os.path.dirname(os.path.abspath(tmp_output_path))
                    qbits_path = os.path.join(qbits_output_dir, 'qbits.npz')
                    np.savez_compressed(qbits_path, qbits=qbits)
                    logger.info(f"Saved qbits allocation to standalone file: {qbits_path}, shape={qbits.shape}, "
                               f"min={qbits.min()}, max={qbits.max()}, mean={qbits.mean():.2f}")

                _t0 = _tick()
                bin_zip_name = bin_dir.split('/')[-1]
                bin_zip_path = os.path.join(exp_dir, f'{bin_zip_name}.zip')
                os.system(f'zip -j {bin_zip_path} {bin_dir}/*')
                _record('zip_pack', _t0)

                zip_file_size = os.path.getsize(bin_zip_path)

                print('final sum:', zip_file_size , 'B')
                print('final sum:', zip_file_size / 1024, 'KB')
                print('final sum:', zip_file_size / 1024 / 1024, 'MB')
                
                with open(bin_zip_path, "rb") as f_in:
                    # 分块读取，避免一次性读入大文件
                    while True:
                        chunk = f_in.read(8192)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                tmp_file.flush()     # 确保写入落盘
                return zip_file_size

    @torch.no_grad
    def decode(self, tmp_file: io.BufferedReader):
        path = tmp_file.name
        print(path)
        with tempfile.TemporaryDirectory() as exp_dir:
            bin_dir = os.path.join(exp_dir, 'bins')
            print('bin_dir', bin_dir)
            os.makedirs(bin_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(bin_dir)
            trans_array = np.load(os.path.join(bin_dir, 't.npz'))["t"]
            
            depth = int(trans_array[0])
            self.depth = depth
            
            # 加载 octree：优先尝试 GPCC 格式，回退到 npz
            oct_gpcc_path = os.path.join(bin_dir, 'oct.gpcc')
            oct_npz_path = os.path.join(bin_dir, 'oct.npz')
            if os.path.exists(oct_gpcc_path):
                logger.info("Loading GPCC-coded octree")
                with open(oct_gpcc_path, 'rb') as f_oct:
                    oct_bitstream = f_oct.read()
                octree, oct_param, _depth = gpcc_decode_octree(oct_bitstream)
                assert _depth == depth, f"GPCC depth mismatch: {_depth} vs {depth}"
                logger.info(f"  GPCC decoded: {len(octree)} points")
            else:
                logger.info("Loading octree from npz (legacy format)")
                oct_vals = np.load(oct_npz_path)
                octree = oct_vals["points"]
                oct_param = oct_vals["params"]
            self.og_number_points = octree.shape[0]

            dxyz, V = decode_oct(oct_param, octree, depth)

            self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
            n_points = dxyz.shape[0]
            
            # 加载 NTK 索引：优先尝试 Laplace 格式，回退到 npz
            ntk_bin_path = os.path.join(bin_dir, 'ntk.bin')
            ntk_npz_path = os.path.join(bin_dir, 'ntk.npz')
            if os.path.exists(ntk_bin_path):
                logger.info("Loading Laplace-coded VQ indices")
                with open(ntk_bin_path, 'rb') as f_ntk:
                    ntk_bitstream = f_ntk.read()
                ntk = ntk_decode(ntk_bitstream)
                logger.info(f"  ntk decoded: {ntk.shape}, range=[{ntk.min()}, {ntk.max()}]")
            else:
                logger.info("Loading VQ indices from npz (legacy format)")
                ntk = np.load(ntk_npz_path)["ntk"]

            # 检查是否有 Laplace 编码的 codebook（新格式）
            cb_q_bin_path = os.path.join(bin_dir, 'cb_q.bin')
            cb_q_npz_path = os.path.join(bin_dir, 'cb_q.npz')
            cb_meta_path = os.path.join(bin_dir, 'cb_meta.npz')
            if os.path.exists(cb_q_bin_path):
                logger.info("Loading Laplace-coded quantized codebook")

                # 先加载元信息（需要知道 cb_quant_bits 来决定解码 dtype）
                cb_meta = np.load(cb_meta_path)
                cb_scales = torch.tensor(cb_meta['scales'], device='cuda')
                cb_zero_points = torch.tensor(cb_meta['zero_points'], device='cuda')
                cb_split = cb_meta['split'].tolist()
                num_keep = int(cb_meta['num_keep'])
                # 读取 cb_quant_bits，兼容旧文件（默认8）
                cb_quant_bits_val = int(cb_meta['cb_quant_bits']) if 'cb_quant_bits' in cb_meta else 8

                # Laplace 解码 codebook 量化索引
                with open(cb_q_bin_path, 'rb') as f_cb:
                    cb_q_bitstream = f_cb.read()
                cb_decode_dtype = np.uint16 if cb_quant_bits_val > 8 else np.uint8
                cb_q_decoded, _, _ = laplace_decode_blocks(cb_q_bitstream, dtype=cb_decode_dtype)
                cb_q_indices = torch.tensor(cb_q_decoded, device='cuda')

                num_entries_from_file = int(cb_meta['num_entries'])
                channels = cb_q_indices.shape[0]
                logger.info(f"  Codebook quantized data: {num_entries_from_file} entries, {channels} channels")
                logger.info(f"  Kept points: {num_keep}, cb_quant_bits: {cb_quant_bits_val}")

                # 反量化整个 codebook
                from .meson_utils import dequantize_kept_sh
                cb = dequantize_kept_sh(
                    cb_q_indices,
                    cb_scales,
                    cb_zero_points,
                    cb_split,
                    num_bits=cb_quant_bits_val
                )

                logger.info(f"  Codebook dequantized: {cb.shape[0]} entries x {cb.shape[1]} dims")
            elif os.path.exists(cb_q_npz_path):
                # 兼容旧格式：npz 存储的量化 codebook
                logger.info("Loading quantized full codebook (legacy npz format)")

                cb_data = np.load(cb_q_npz_path)
                cb_q_indices = torch.tensor(cb_data['q_indices'], device='cuda')
                cb_scales = torch.tensor(cb_data['scales'], device='cuda')
                cb_zero_points = torch.tensor(cb_data['zero_points'], device='cuda')
                cb_split = cb_data['split'].tolist()
                num_keep = int(cb_data['num_keep'])
                # 读取 cb_quant_bits，兼容旧文件（默认8）
                cb_quant_bits_val = int(cb_data['cb_quant_bits']) if 'cb_quant_bits' in cb_data else 8

                num_entries_from_file = int(cb_data['num_entries'])
                channels = cb_q_indices.shape[0]
                logger.info(f"  Codebook quantized data: {num_entries_from_file} entries, {channels} channels")
                logger.info(f"  Kept points: {num_keep}, cb_quant_bits: {cb_quant_bits_val}")

                from .meson_utils import dequantize_kept_sh
                cb = dequantize_kept_sh(
                    cb_q_indices,
                    cb_scales,
                    cb_zero_points,
                    cb_split,
                    num_bits=cb_quant_bits_val
                )

                logger.info(f"  Codebook dequantized: {cb.shape[0]} entries x {cb.shape[1]} dims")
            else:
                # 原始模式：直接加载完整 codebook（float32）
                cb = torch.tensor(np.load(os.path.join(bin_dir , 'um.npz'))["umap"], device='cuda')
                logger.info(f"Loading standard VQ codebook: {cb.shape[0]} entries")
            
            # print('ntk.shape', ntk.shape)
            # print('cb.shape', cb.shape)

            # 检查索引范围，防止越界
            ntk_min, ntk_max = ntk.min(), ntk.max()
            logger.info(f"Feature indices range: [{ntk_min}, {ntk_max}], codebook size: {cb.shape[0]}")
            if ntk_max >= cb.shape[0]:
                logger.error(f"Index out of bounds! Max index {ntk_max} >= codebook size {cb.shape[0]}")
                raise ValueError(f"Feature indices contain out-of-bounds values: max={ntk_max}, codebook_size={cb.shape[0]}")

            features_rest = torch.zeros([ntk.shape[0], cb.shape[1]], device='cuda')
            for i in range(ntk.shape[0]):
                features_rest[i] = cb[int(ntk[i])]
            self.n_sh = (self.max_sh_degree + 1) ** 2
            self._features_rest = nn.Parameter(features_rest).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False)
            
            # self._features_rest = nn.Parameter(
            #     torch.matmul(
            #         F.one_hot(torch.tensor(ntk, dtype=torch.long, device="cuda")).float(), 
            #         torch.tensor(cb, dtype=torch.float, device="cuda")
            #     ).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False))
            
            # print('gaussian model, line 1027, trans_array', trans_array.shape, trans_array)
            
            # 解码 orgb（RAHT AC 系数量化索引）
            orgb_bin_path = os.path.join(bin_dir, 'orgb.bin')
            orgb_npz_path = os.path.join(bin_dir, 'orgb.npz')
            if os.path.exists(orgb_bin_path):
                # 新格式: Laplace 解码
                with open(orgb_bin_path, 'rb') as f_orgb:
                    orgb_bitstream = f_orgb.read()
                orgb_decoded, _, _ = laplace_decode_blocks(orgb_bitstream)  # auto dtype from header
                # orgb_decoded: [channels, N], 需要转置为 [N, channels]
                q_orgb_i = torch.tensor(orgb_decoded.astype(np.float32), dtype=torch.float, device="cuda").transpose(0, 1)
                
                # DC 系数单独存储
                orgb_dc_path = os.path.join(bin_dir, 'orgb_dc.npz')
                orgb_f = torch.tensor(np.load(orgb_dc_path)["f"], dtype=torch.float, device="cuda")
                logger.info(f"  orgb decoded from Laplace: {q_orgb_i.shape}")
            else:
                # 兼容旧格式: npz 存储
                oef_vals = np.load(orgb_npz_path)
                orgb_f = torch.tensor(oef_vals["f"], dtype=torch.float, device="cuda")
                q_dtype = get_dtype_for_bits(self.num_bits)
                q_orgb_i = torch.tensor(oef_vals["i"].astype(q_dtype), dtype=torch.float, device="cuda").reshape(self.n_rfc - 3, -1).contiguous().transpose(0, 1)
            
            # 解码 ct（scaling 量化索引）
            ct_bin_path = os.path.join(bin_dir, 'ct.bin')
            ct_npz_path = os.path.join(bin_dir, 'ct.npz')
            if os.path.exists(ct_bin_path):
                # 新格式: Laplace 解码
                with open(ct_bin_path, 'rb') as f_ct:
                    ct_bitstream = f_ct.read()
                ct_decoded, _, _ = laplace_decode_blocks(ct_bitstream)  # auto dtype from header
                # ct_decoded: [3, N], 转置为 [N, 3]
                q_scale_i = torch.tensor(ct_decoded.astype(np.float32), dtype=torch.float, device="cuda").transpose(0, 1)
                logger.info(f"  ct decoded from Laplace: {q_scale_i.shape}")
            else:
                # 兼容旧格式: npz 存储
                q_dtype = get_dtype_for_bits(self.num_bits)
                q_scale_i = torch.tensor(np.load(ct_npz_path)["i"].astype(q_dtype), dtype=torch.float, device="cuda").reshape(3, -1).contiguous().transpose(0, 1)
            
            print('rf_orgb_f size', orgb_f.shape)
            print('q_rf_orgb_i.shape', q_orgb_i.shape)
            print('q_scale_i.shape', q_scale_i.shape)
            
            # lseg = int(trans_array[1])
            # self.lseg = lseg
            n_block = int(trans_array[1])
            self.n_block = n_block
            
            '''dequant'''
            qa_cnt = 2
            rf_orgb = []
            rf_len = q_orgb_i.shape[0]
            # print('rf_len, n_points', rf_len, n_points)
            assert rf_len + 1 == n_points
            # if rf_len % self.lseg == 0:
            #     n_rf = rf_len // self.lseg
            # else:
            #     n_rf = rf_len // self.lseg + 1
            split = split_length(rf_len, n_block)
            for i in range(self.n_rfc - 3):
                rf_i = torch_vanilla_dequant_ave(q_orgb_i[:, i], split, trans_array[qa_cnt:qa_cnt+2*n_block])
                # print('rf_i.shape', rf_i.shape)
                rf_orgb.append(rf_i.reshape(-1, 1))
                qa_cnt += 2*n_block
            rf_orgb = torch.concat(rf_orgb, dim=-1)
            
            
            de_scale = []
            scale_len = q_scale_i.shape[0]
            assert scale_len == n_points
            # if scale_len % self.lseg == 0:
            #     n_scale = scale_len // self.lseg
            # else:
            #     n_scale = scale_len // self.lseg + 1
            scale_split = split_length(scale_len, n_block)
            for i in range(3):
                scale_i = torch_vanilla_dequant_ave(q_scale_i[:, i], scale_split, trans_array[qa_cnt:qa_cnt+2*n_block])
                de_scale.append(scale_i.reshape(-1, 1))
                qa_cnt += 2*n_block
            de_scale = torch.concat(de_scale, axis=-1).to("cuda")
            self._scaling = nn.Parameter(de_scale.requires_grad_(False))
            
            print('qa_cnt', qa_cnt, 'trans', trans_array.shape)
            print('rf_orgb.shape, de_scale.shape', rf_orgb.shape, de_scale.shape)
            
            C = torch.concat([orgb_f.reshape(1, -1), rf_orgb], 0)
            
            w, val, reorder = copyAsort(V)

            # print('saving 2')
            self.reorder = reorder  
            
            if self.raht:
                res_inv = inv_haar3D_param(V, depth)
                pos = res_inv['pos']
                iW1 = res_inv['iW1']
                iW2 = res_inv['iW2']
                iS = res_inv['iS']
                
                iLeft_idx = res_inv['iLeft_idx']
                iRight_idx = res_inv['iRight_idx']
            
                iLeft_idx_CT = res_inv['iLeft_idx_CT']
                iRight_idx_CT = res_inv['iRight_idx_CT']
                iTrans_idx = res_inv['iTrans_idx']
                iTrans_idx_CT = res_inv['iTrans_idx_CT']

                CT_yuv_q_temp = C[pos.astype(int)]
                raht_features = torch.zeros(C.shape).cuda()
                OC = torch.zeros(C.shape).cuda()

                for i in range(depth*3):
                    w1 = iW1[i]
                    w2 = iW2[i]
                    S = iS[i]
                    
                    left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
                    left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
                    
                    trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
                    
                    
                    OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
                    OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                            w2, 
                                                            CT_yuv_q_temp[left_idx_CT], 
                                                            CT_yuv_q_temp[right_idx_CT])  
                    CT_yuv_q_temp[:S] = OC[:S]

                raht_features[reorder] = OC
            else:
                logger.info("RAHT ablation: skipping RAHT inverse transform in decode")
                raht_features = torch.zeros(C.shape).cuda()
                raht_features[reorder] = C
            
            # rf_chk = torch.tensor(np.load('/home/szxie/mesongs/duipai/rf_cpu.npy'), dtype=torch.float, device='cuda')
            # print(torch.sum(torch.square(rf_chk - raht_features)))
            
            # scale_chk = torch.tensor(np.load('/home/szxie/mesongs/duipai/scale_cpu.npy'), dtype=torch.float, device='cuda')
            # print(torch.sum(torch.square(scale_chk - de_scale)))
            
            self._opacity = nn.Parameter(raht_features[:, :1].requires_grad_(False))
            if self.use_quat:
                self._rotation = nn.Parameter(raht_features[:, 1:5].requires_grad_(False))
                self._features_dc = nn.Parameter(raht_features[:, 5:].unsqueeze(1).requires_grad_(False))
            else:
                self._euler = nn.Parameter(raht_features[:, 1:4].nan_to_num_(0).requires_grad_(False))
                self._features_dc = nn.Parameter(raht_features[:, 4:].unsqueeze(1).requires_grad_(False))
            # print('max euler', torch.max(self._euler))
            self.active_sh_degree = self.max_sh_degree