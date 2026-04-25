from dataclasses import dataclass, fields, asdict
import math
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task


from .config import RDOGaussianModelParams, RDOGaussianOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_covariance_from_scaling_rotation, build_rotation
)
from splatwizard.rasterizer.gaussian import GaussianRasterizationSettings, GaussianRasterizer
from ..._cmod.simple_knn import distCUDA2
from ...compression.vq.ecvq import ECVQ
from ...modules.dataclass import RenderResult, ModelContext
from ...utils.graphics_utils import BasicPointCloud
from ...utils.sh_utils import RGB2SH, eval_sh


index_cache_global_dict = {
    'scale': None,
    'rot': None,
    'dc': None,
    'sh1': None,
    'sh2': None,
    'sh3': None
}

@dataclass
class RDOGaussianVQModelParams:
    scale: int
    rot: int
    dc: int
    sh1: int
    sh2: int
    sh3: int


@dataclass
class RDOGaussianVQModelOptParams:
    scale: int
    rot: int
    dc: int
    sh1: int
    sh2: int
    sh3: int


@dataclass
class RODContext(ModelContext):
    activate_vq: bool = False
    activate_gsprune: bool = False
    activate_shprune: bool = False


@dataclass
class RDORenderResult(RenderResult):
    rate_loss: torch.Tensor = None
    vq_loss: torch.Tensor = None
    sh_mask_loss: torch.Tensor = None
    sh_mask_percent: torch.Tensor = None
    gs_mask_loss: torch.Tensor = None
    gs_mask_percent: torch.Tensor = None
    bits: Any = None


def build_vq_model(params: RDOGaussianModelParams):
    return RDOGaussianVQModelParams(
        scale=params.vq_scale_cbsize,
        rot=params.vq_rot_cbsize,
        dc=params.vq_dc_cbsize,
        sh1=params.vq_sh1_cbsize,
        sh2=params.vq_sh2_cbsize,
        sh3=params.vq_sh3_cbsize,
        # patch_size=params.vq_patch_size,
    )


def build_vq_opt(opt: RDOGaussianOptimizationParams):
    return RDOGaussianVQModelOptParams(
        scale=opt.vq_scale_lmbda,
        rot=opt.vq_rot_lmbda,
        dc=opt.vq_dc_lmbda,
        sh1=opt.vq_sh1_lmbda,
        sh2=opt.vq_sh2_lmbda,
        sh3=opt.vq_sh3_lmbda,
    )


class RDOGaussian(GaussianModel):
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, model_param: RDOGaussianModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.vq_cfg = build_vq_model(model_param)
        self.vq_patch_size = model_param.vq_patch_size
        self.gs_mask_thres = 0.1
        self.sh_mask_thres = 0.1

        self._context = RODContext()

        self._quantizer = nn.ModuleDict()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: RDOGaussianOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

        scheduler.register_task(opt.vq_start_iter, task=self.activate_vq)
        scheduler.register_task(opt.rate_constrain_iter, task=self.enable_rate_constraint)
        scheduler.register_task(opt.gs_prune_start_iter, task=self.activate_gsprune)
        scheduler.register_task(opt.sh_prune_start_iter, task=self.activate_shprune)

        # post task but after optimizer_step, transform as pre task
        scheduler.register_task(range(opt.vq_start_iter + 1, opt.rate_constrain_iter + 1, opt.reactivate_codeword_period),
                                task=self.reactivate_codeword)

        scheduler.register_task(range(opt.densify_until_iter + 1), task=self.add_densification_stats)

        scheduler.register_task(
            range(opt.densify_from_iter + 1, opt.densify_until_iter + 1, opt.densification_interval),
            task=self.densify_and_prune_task, logging=True
        )

        if ppl.white_background:
            scheduler.register_task(opt.densify_from_iter + 1, task=self.reset_opacity, logging=True)
        scheduler.register_task(range(1, opt.densify_until_iter + 1, opt.opacity_reset_interval),
                                task=self.reset_opacity, logging=True)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: RDOGaussianOptimizationParams):
        pass

    @task
    def activate_vq(self):
        self._context.activate_vq = True
        for key in asdict(self.vq_cfg).keys():
            self._quantizer[key].kmeans_initialize(self.vector_dict[key].data)

    @task
    def enable_rate_constraint(self):
        assert self._context.activate_vq
        # print(f"\n[ITER {iteration}] Set rate_cfor key in asdict(self.vq_cfg).keys():
        for key in asdict(self.vq_cfg).keys():
            self._quantizer[key].rate_constrain = True

    @task
    def activate_gsprune(self):
        self._context.activate_gsprune = True

    @task
    def activate_shprune(self):
        self._context.activate_shprune = True

    @task
    def reactivate_codeword(self):
        for key in asdict(self.vq_cfg).keys():
            self._quantizer[key].reactivate_codeword(prob_threshold=1e-5)

    def optimizer_step(self, render_result: RenderResult, opt: OptimizationParams, step: int):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self._context.activate_vq:
            self.optimizer_vq.step()
            self.scheduler_vq.step()
        self.optimizer_vq.zero_grad(set_to_none=True)
        if self._context.activate_shprune:
            self.optimizer_sh_mask.step()
        self.optimizer_sh_mask.zero_grad(set_to_none=True)
        if self._context.activate_gsprune:
            self.optimizer_gs_mask.step()
        self.optimizer_gs_mask.zero_grad(set_to_none=True)

    def render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams=None, step=0, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        update_index = (step - 1) % 10 == 0
        vq_opt = asdict(opt) if opt else None
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        num_gs = means3D.shape[0]

        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.scaling
            rotations = self.rotation

        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = self.features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        if self._context.activate_gsprune:
            # activated scale and opacity
            gs_prune_result_dict = self.apply_gs_mask()
            scales = gs_prune_result_dict['scale']
            opacity = gs_prune_result_dict['opacity']
            gs_soft_mask = gs_prune_result_dict['gs_soft_mask']
            gs_hard_mask = gs_prune_result_dict['gs_hard_mask']
            gs_bitmask = gs_prune_result_dict['gs_bitmask']
            gs_mask_loss = torch.mean(gs_soft_mask)
            # calculate mask percentage
            gs_mask_percent = 1 - gs_hard_mask.sum() / gs_hard_mask.numel()
        else:
            gs_mask_loss = torch.tensor(0)
            gs_mask_percent = 0

        if self._context.activate_shprune:
            sh_prune_result_dict = self.apply_sh_mask()
            features_rest = sh_prune_result_dict['features_rest']
            sh_soft_mask = sh_prune_result_dict['sh_soft_mask']
            sh_hard_mask = sh_prune_result_dict['sh_hard_mask']
            sh1_bitmask = sh_prune_result_dict['sh1_bitmask']
            sh2_bitmask = sh_prune_result_dict['sh2_bitmask']
            sh3_bitmask = sh_prune_result_dict['sh3_bitmask']
            if self._context.activate_gsprune:
                sh_hard_mask = sh_hard_mask[gs_bitmask]
                sh_soft_mask = sh_soft_mask[gs_bitmask]
                sh1_bitmask = torch.logical_and(sh1_bitmask, gs_bitmask)
                sh2_bitmask = torch.logical_and(sh2_bitmask, gs_bitmask)
                sh3_bitmask = torch.logical_and(sh3_bitmask, gs_bitmask)
            sh_mask_loss = torch.mean(sh_soft_mask)
            # calculate mask percentage
            sh_mask_percent = 1 - sh_hard_mask.sum() / sh_hard_mask.numel()
            shs[:, 1:, :] = features_rest
        else:
            sh_mask_loss = torch.tensor(0)
            sh_mask_percent = 0

        def ste(y_hat, y):
            return (y_hat - y).detach() + y

        rate_loss = [torch.tensor(0)]
        vq_loss = [torch.tensor(0)]
        bits_dict = {}
        index_cache_dict = {}
        vq_dim = 0

        if self._context.activate_vq:
            assert vq_opt
            vq_inputs = self.vector_dict
            vq_out = {}
            vq_inputs['sh1'] = features_rest[sh1_bitmask, :3, :].flatten(start_dim=1)
            vq_inputs['sh2'] = features_rest[sh2_bitmask, 3:8, :].flatten(start_dim=1)
            vq_inputs['sh3'] = features_rest[sh3_bitmask, 8:, :].flatten(start_dim=1)
            vq_inputs['scale'] = scales[gs_bitmask, :]
            vq_inputs['rot'] = vq_inputs['rot'][gs_bitmask, :]
            vq_inputs['dc'] = vq_inputs['dc'][gs_bitmask, :]
            if not update_index:
                # use cached indexes
                index_cache_dict['scale'] = index_cache_global_dict['scale'][gs_bitmask]
                index_cache_dict['rot'] = index_cache_global_dict['rot'][gs_bitmask]
                index_cache_dict['dc'] = index_cache_global_dict['dc'][gs_bitmask]
                index_cache_dict['sh1'] = index_cache_global_dict['sh1'][sh1_bitmask]
                index_cache_dict['sh2'] = index_cache_global_dict['sh2'][sh2_bitmask]
                index_cache_dict['sh3'] = index_cache_global_dict['sh3'][sh3_bitmask]

            for vq_key in asdict(self.vq_cfg):
                result = self._quantizer[vq_key](vq_inputs[vq_key],
                                               index_cache=None if update_index else index_cache_dict[vq_key])
                vq_out[vq_key] = ste(y_hat=result['x_hat'], y=vq_inputs[vq_key])
                vq_out[vq_key] = self.vq_post_process(vq_key, vq_out[vq_key])
                bits_dict[vq_key] = result['bits']
                rate_loss.append(bits_dict[vq_key] / vq_opt[vq_key])
                index_cache_dict[vq_key] = result['x_index']
                vq_loss.append(torch.sum(torch.norm(result['x_hat'] - vq_inputs[vq_key], dim=-1)))
                vq_dim += vq_inputs[vq_key].shape[-1]

            scales[gs_bitmask, :] = vq_out['scale']
            rotations[gs_bitmask, :] = vq_out['rot']
            shs[gs_bitmask, 0:1, :] = vq_out['dc']
            shs[sh1_bitmask, 1:4, :] = vq_out['sh1']
            shs[sh2_bitmask, 4:9, :] = vq_out['sh2']
            shs[sh3_bitmask, 9:, :] = vq_out['sh3']

            if index_cache_global_dict['scale'] is None:
                for key in index_cache_global_dict.keys():
                    index_cache_global_dict[key] = torch.zeros(num_gs, 1, device='cuda', dtype=torch.long)

            index_cache_global_dict['scale'][gs_bitmask] = index_cache_dict['scale']
            index_cache_global_dict['rot'][gs_bitmask] = index_cache_dict['rot']
            index_cache_global_dict['dc'][gs_bitmask] = index_cache_dict['dc']
            index_cache_global_dict['sh1'][sh1_bitmask] = index_cache_dict['sh1']
            index_cache_global_dict['sh2'][sh2_bitmask] = index_cache_dict['sh2']
            index_cache_global_dict['sh3'][sh3_bitmask] = index_cache_dict['sh3']

        if vq_dim != 0:
            rate_loss = sum(rate_loss) / num_gs / vq_dim
            vq_loss = sum(vq_loss) / num_gs / vq_dim
        else:
            rate_loss = sum(rate_loss) / num_gs
            vq_loss = sum(vq_loss) / num_gs

        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )

        # return {
        #     "render": rendered_image,
        #     "viewspace_points": screenspace_points,
        #     "visibility_filter": radii > 0,
        #     "radii": radii,
        #     "rate_loss": rate_loss,
        #     "vq_loss": vq_loss,
        #     "sh_mask_loss": sh_mask_loss,
        #     "sh_mask_percent": sh_mask_percent,
        #     "gs_mask_loss": gs_mask_loss,
        #     "gs_mask_percent": gs_mask_percent,
        #     "bits": bits_dict
        # }

        return RDORenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii,
            rate_loss=rate_loss,
            vq_loss=vq_loss,
            sh_mask_loss=sh_mask_loss,
            sh_mask_percent=sh_mask_percent,
            gs_mask_loss=gs_mask_loss,
            gs_mask_percent=gs_mask_percent,
            bits=bits_dict
        )

    @property
    def vector_dict(self):
        return {
            'scale': self.scaling.flatten(start_dim=1),
            'rot': self._rotation.flatten(start_dim=1),
            'dc': self._features_dc.flatten(start_dim=1),
            'sh1': self._features_rest.tensor_split((3, 8), dim=1)[0].flatten(start_dim=1),
            'sh2': self._features_rest.tensor_split((3, 8), dim=1)[1].flatten(start_dim=1),
            'sh3': self._features_rest.tensor_split((3, 8), dim=1)[2].flatten(start_dim=1)
        }

    def vq_post_process(self, key, value):
        if key == 'scale':
            return value
        if key == 'rot':
            return self.rotation_activation(value)
        if key == 'dc':
            return torch.unsqueeze(value, 1)
        if key == 'sh1':
            return torch.reshape(value, (-1, 3, 3))
        if key == 'sh2':
            return torch.reshape(value, (-1, 5, 3))
        if key == 'sh3':
            return torch.reshape(value, (-1, 7, 3))

    def initialize_quantizer(self, opt: RDOGaussianOptimizationParams):
        vq_cfg = self.vq_cfg
        x_dim = {
            'scale': self._scaling.flatten(start_dim=1).shape[-1],
            'rot': self._rotation.flatten(start_dim=1).shape[-1],
            'dc': self._features_dc.flatten(start_dim=1).shape[-1],
            'sh1': self._features_rest.tensor_split((3, 8), dim=1)[0].flatten(start_dim=1).shape[-1],
            'sh2': self._features_rest.tensor_split((3, 8), dim=1)[1].flatten(start_dim=1).shape[-1],
            'sh3': self._features_rest.tensor_split((3, 8), dim=1)[2].flatten(start_dim=1).shape[-1]
        }

        vq_opt = build_vq_opt(opt)
        vq_opt = asdict(vq_opt)
        for key, cb_size in asdict(vq_cfg):

            self._quantizer[key] = ECVQ(
                x_dim=x_dim[key],
                cb_dim=x_dim[key],
                cb_size=cb_size,
                lmbda=vq_opt[key],
                patch_size=self.vq_patch_size,
                rate_constrain=False,
            ).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._context
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._context) = model_args
        self.sh_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 1, device="cuda").requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def xyz(self):
        return self._xyz

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, cam_info=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logger.info("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")
        self.sh_mask = nn.Parameter(torch.zeros(features.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(features.shape[0], 1, device="cuda").requires_grad_(True))

    def training_setup(self, training_args: RDOGaussianOptimizationParams):
        self.initialize_quantizer(training_args)

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        l = [
            {'params': [self._quantizer[key].codebook for key in self._quantizer.keys()], 'lr': training_args.vq_cb_lr},
            {'params': [self._quantizer[key].logits for key in self._quantizer.keys()], 'lr': training_args.vq_logits_lr}
        ]
        self.optimizer_vq = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_vq = MultiStepLR(self.optimizer_vq, milestones=[12000], gamma=0.1)
        l = [
            {'params': [self.sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"}
        ]
        self.optimizer_sh_mask = torch.optim.Adam(l, lr=training_args.sh_mask_lr, eps=1e-15)
        l = [
            {'params': [self.gs_mask], 'lr': training_args.gs_mask_lr, "name": "gs_mask"}
        ]
        self.optimizer_gs_mask = torch.optim.Adam(l, lr=training_args.gs_mask_lr, eps=1e-15)

    @task
    def update_learning_rate(self, iteration: int):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    # def apply_and_save_vq(self, vq_path):
    #     mkdir_p(vq_path)
    #
    #     # Remove pruned Gaussians
    #     gs_bitmask = self.apply_gs_mask()['gs_bitmask']
    #     self.prune_points(~gs_bitmask)
    #     sh_bitmask = {}
    #     sh_prune_result_dict = self.apply_sh_mask()
    #     for sh_key in ['sh1', 'sh2', 'sh3']:
    #         sh_bitmask[sh_key] = sh_prune_result_dict[f'{sh_key}_bitmask'].detach().contiguous().cpu().numpy()
    #
    #     # Apply VQ
    #     num_gaussians = self._xyz.shape[0]
    #     vq_out = {}
    #     vq_indexes = {}
    #     vq_codebooks = {}
    #     vq_logits = {}
    #     remove_negatives(self._quantizer['scale'].codebook)
    #     for vq_key in self.vq_cfg['keys']:
    #         if vq_key in ['sh1', 'sh2', 'sh3']:
    #             vq_result = self._quantizer[vq_key](self.vector_dict[vq_key][sh_bitmask[vq_key]])
    #         else:
    #             vq_result = self._quantizer[vq_key](self.vector_dict[vq_key])
    #         vq_xhat = self.vq_post_process(vq_key, vq_result["x_hat"]) if vq_key != 'rot' else vq_result["x_hat"]
    #         vq_out[vq_key] = vq_xhat
    #         vq_indexes[vq_key] = vq_result['x_index'].detach().contiguous().cpu().numpy()
    #         vq_codebooks[vq_key] = self._quantizer[vq_key].codebook.detach().contiguous().cpu().numpy()
    #         vq_logits[vq_key] = self._quantizer[vq_key].logits.detach().contiguous().cpu().numpy()
    #
    #     features_rest = torch.zeros_like(self._features_rest.data)
    #     features_rest[sh_bitmask['sh1'], :3, :] = vq_out['sh1']
    #     features_rest[sh_bitmask['sh2'], 3:8, :] = vq_out['sh2']
    #     features_rest[sh_bitmask['sh3'], 8:, :] = vq_out['sh3']
    #     self._features_rest.data = features_rest
    #     self._features_dc.data = vq_out['dc']
    #     self._scaling.data = self.scaling_inverse_activation(vq_out['scale'])
    #     self._rotation.data = vq_out['rot']
    #
    #     # Rearrange Gaussians by SH masks
    #     sort_idx, boundaries = shmask_sort(sh_bitmask)
    #     sh = -np.ones((num_gaussians, 3), dtype=int)
    #     sh[sh_bitmask['sh1'], 0] = vq_indexes['sh1'].squeeze()
    #     sh[sh_bitmask['sh2'], 1] = vq_indexes['sh2'].squeeze()
    #     sh[sh_bitmask['sh3'], 2] = vq_indexes['sh3'].squeeze()
    #
    #     sh = sh[sort_idx]
    #
    #     vq_indexes['sh1'] = sh[boundaries[3]:, 0:1]
    #     vq_indexes['sh2'] = np.concatenate([sh[boundaries[1]:boundaries[3], 1:2],
    #                                         sh[boundaries[5]:, 1:2]])
    #     vq_indexes['sh3'] = np.concatenate([sh[boundaries[0]:boundaries[1], 2:],
    #                                         sh[boundaries[2]:boundaries[3], 2:],
    #                                         sh[boundaries[4]:boundaries[5], 2:],
    #                                         sh[boundaries[6]:, 2:]])
    #     for key in ['scale', 'rot', 'dc']:
    #         vq_indexes[key] = vq_indexes[key][sort_idx]
    #     vq_codebooks, vq_logits, vq_indexes = shrink_codebook(vq_codebooks, vq_logits, vq_indexes)
    #
    #     # Non-VQ attributes
    #     xyz = self._xyz.detach().cpu().numpy().astype(np.float16)[sort_idx]
    #     opacities = self._opacity.detach().cpu().numpy()[sort_idx]
    #     quantized_opacities, opacities, opacity_log_prob, step_size, min_opacity = opacity_quant(opacities)
    #     vq_logits['opa'] = opacity_log_prob
    #
    #     # Arithmetic coding
    #     index_strings = []
    #     index_lengths = []
    #
    #     for vq_key in vq_indexes.keys():
    #         bits_str = entropy_coding(torch.Tensor(vq_indexes[vq_key]), torch.Tensor(vq_logits[vq_key]))
    #         index_strings.append(bits_str)
    #         index_lengths.append(len(bits_str))
    #
    #     bits_str = entropy_coding(torch.Tensor(quantized_opacities), torch.Tensor(opacity_log_prob))
    #     index_strings.append(bits_str)
    #     index_lengths.append(len(bits_str))
    #
    #     index_bitstream = pack_strings(index_strings)
    #     boundaries_bitstream = pack_uints(boundaries.tolist())
    #     opacity_header_bitstream = pack_floats([step_size, min_opacity])
    #
    #     with open(os.path.join(vq_path, 'index_bitstream.bin'), "wb") as f:
    #         f.write(index_bitstream)
    #     with open(os.path.join(vq_path, 'header.bin'), "wb") as f:
    #         f.write(boundaries_bitstream)
    #         f.write(opacity_header_bitstream)
    #
    #     np.savez(os.path.join(vq_path, 'codebook.npz'), **vq_codebooks)
    #     np.savez(os.path.join(vq_path, 'logits.npz'), **vq_logits)
    #     np.savez(os.path.join(vq_path, 'position.npz'), position=xyz)

    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))
    #
    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()
    #
    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    #
    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)
    #
    # def reset_opacity(self):
    #     opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]
    #
    # def decode(self, bitstream_path):
    #     xyz = np.load(os.path.join(bitstream_path, 'position.npz'))['position']
    #     vq_codebooks = np.load(os.path.join(bitstream_path, 'codebook.npz'))
    #     logits = np.load(os.path.join(bitstream_path, 'logits.npz'))
    #     boundaries, step_size, min_opacity = decode_header(os.path.join(bitstream_path, 'header.bin'))
    #     shape_dict = get_index_shapes(boundaries)
    #     index_path = os.path.join(bitstream_path, 'index_bitstream.bin')
    #     indexes = decode_indexes(index_path, logits, shape_dict)
    #     sh_bitmask = get_sh_bitmask(boundaries)
    #     opacities = opacity_dequant(indexes.pop('opa'), min_opacity, step_size).to('cuda')
    #
    #     vq_attributes = {}
    #
    #     for vq_key in vq_codebooks.files:
    #         codebook = vq_codebooks[vq_key]
    #         index = indexes[vq_key].squeeze(1)
    #         attr = codebook[0, index, :]
    #         attr = torch.tensor(attr, dtype=torch.float, device="cuda")
    #         vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['rot', 'scale'] else attr
    #
    #     num_gaussians = boundaries[-1]
    #     features_dc = vq_attributes['dc']
    #     features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")
    #     features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
    #     features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
    #     features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']
    #     scales = self.scaling_inverse_activation(vq_attributes['scale'])
    #     rots = vq_attributes['rot']
    #
    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
    #     self._features_dc = nn.Parameter(features_dc)
    #     self._features_rest = nn.Parameter(features_extra)
    #     self._opacity = nn.Parameter(opacities)
    #     self._scaling = nn.Parameter(scales)
    #     self._rotation = nn.Parameter(rots)
    #
    #     self.active_sh_degree = self.max_sh_degree
    #
    # def load_vq(self, vq_path):
    #     xyz = np.load(os.path.join(vq_path, 'position.npz'))['position']
    #     opacities = np.load(os.path.join(vq_path, 'opacity.npz'))['opacity']
    #     num_gaussians = xyz.shape[0]
    #     vq_codebooks = np.load(os.path.join(vq_path, 'codebook.npz'))
    #     vq_indexes = np.load(os.path.join(vq_path, 'index.npz'))
    #     if os.path.exists(os.path.join(vq_path, 'sh_bitmask.npz')):
    #         sh_bitmask = np.load(os.path.join(vq_path, 'sh_bitmask.npz'))
    #         sh_bitmask = {sh_key: sh_bitmask[sh_key] for sh_key in ['sh1', 'sh2', 'sh3']}
    #     elif os.path.exists(os.path.join(vq_path, 'header.npz')):
    #         boundaries = np.load(os.path.join(vq_path, 'header.npz'))['boundaries']
    #         shmask_sorted = np.zeros(num_gaussians, dtype=int)
    #         for i in range(7, -1, -1):
    #             shmask_sorted[:boundaries[i]] = i
    #
    #         sh_bitmask = {sh_key: np.zeros(num_gaussians, dtype=bool) for sh_key in ['sh1', 'sh2', 'sh3']}
    #
    #         sh_bitmask['sh1'] |= (shmask_sorted & (1 << 2)).astype(bool)
    #         sh_bitmask['sh2'] |= (shmask_sorted & (1 << 1)).astype(bool)
    #         sh_bitmask['sh3'] |= (shmask_sorted & (1 << 0)).astype(bool)
    #     else:
    #         sh_bitmask = {sh_key: np.ones((num_gaussians,), dtype=np.bool_) for sh_key in ['sh1', 'sh2', 'sh3']}
    #     vq_attributes = {}
    #
    #     for vq_key in vq_codebooks.files:
    #         codebook = vq_codebooks[vq_key]
    #         index = vq_indexes[vq_key].squeeze(1)
    #         attr = codebook[0, index, :]
    #         attr = torch.tensor(attr, dtype=torch.float, device="cuda")
    #         vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['rot', 'scale'] else attr
    #
    #     features_dc = vq_attributes['dc']
    #     features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")
    #     features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
    #     features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
    #     features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']
    #     scales = self.scaling_inverse_activation(vq_attributes['scale'])
    #     rots = vq_attributes['rot']
    #
    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
    #     self._features_dc = nn.Parameter(features_dc)
    #     self._features_rest = nn.Parameter(features_extra)
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
    #     self._scaling = nn.Parameter(scales)
    #     self._rotation = nn.Parameter(rots)
    #
    #     self.active_sh_degree = self.max_sh_degree
    #
    # def load_ply(self, path):
    #     plydata = PlyData.read(path)
    #
    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])), axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    #
    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    #
    #     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    #     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
    #
    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #
    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #
    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._features_dc = nn.Parameter(
    #         torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
    #             True))
    #     self._features_rest = nn.Parameter(
    #         torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
    #             True))
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self.sh_mask = nn.Parameter(torch.zeros(xyz.shape[0], 3, device="cuda").requires_grad_(True))
    #     self.gs_mask = nn.Parameter(torch.zeros(xyz.shape[0], 1, device="cuda").requires_grad_(True))
    #
    #     self.active_sh_degree = self.max_sh_degree
    #
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_gs_mask.param_groups:
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_sh_mask.param_groups:
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        for group in self.optimizer_gs_mask.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        for group in self.optimizer_sh_mask.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_gs_mask, new_sh_mask):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "gs_mask": new_gs_mask,
             "sh_mask": new_sh_mask
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_gs_mask = self.gs_mask[selected_pts_mask].repeat(N, 1)
        new_sh_mask = self.sh_mask[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_gs_mask, new_sh_mask)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_gs_mask = self.gs_mask[selected_pts_mask]
        new_sh_mask = self.sh_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_gs_mask, new_sh_mask)

    @task
    def densify_and_prune_task(self, opt: RDOGaussianOptimizationParams, step: int):
        max_screen_size = 20 if step > opt.opacity_reset_interval else None
        max_grad = opt.densify_grad_threshold
        min_opacity = 0.005
        extent = self.spatial_lr_scale

        # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    @task
    def add_densification_stats(self, render_result: RenderResult):
        visibility_filter = render_result.visibility_filter
        radii = render_result.radii
        update_filter = visibility_filter
        viewspace_point_tensor = render_result.viewspace_points
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def apply_sh_mask(self):
        sh_soft_mask = torch.sigmoid(torch.cat([self.sh_mask[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
                                                self.sh_mask[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
                                                self.sh_mask[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1))
        sh_hard_mask = ((sh_soft_mask > self.sh_mask_thres).float() - sh_soft_mask).detach() + sh_soft_mask
        features_rest = torch.mul(self._features_rest, sh_hard_mask)

        return {
            'features_rest': features_rest,
            'sh_soft_mask': sh_soft_mask,
            'sh_hard_mask': sh_hard_mask,
            'sh1_bitmask': torch.sigmoid(self.sh_mask[:, 0]) > self.sh_mask_thres,
            'sh2_bitmask': torch.sigmoid(self.sh_mask[:, 1]) > self.sh_mask_thres,
            'sh3_bitmask': torch.sigmoid(self.sh_mask[:, 2]) > self.sh_mask_thres,
        }

    def apply_gs_mask(self):
        gs_soft_mask = torch.sigmoid(self.gs_mask)
        gs_hard_mask = ((gs_soft_mask > self.gs_mask_thres).float() - gs_soft_mask).detach() + gs_soft_mask

        scales = torch.mul(self.scaling, gs_hard_mask)
        opacity = torch.mul(self.opacity, gs_hard_mask)
        return {
            'scale': scales,
            'opacity': opacity,
            'gs_soft_mask': gs_soft_mask,
            'gs_hard_mask': gs_hard_mask,
            'gs_bitmask': gs_soft_mask.flatten() > self.gs_mask_thres
        }

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]



