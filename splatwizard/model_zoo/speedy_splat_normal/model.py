import math
import typing
from dataclasses import dataclass

import torch
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task
from splatwizard.rasterizer.speedy import SpeedyGaussianRasterizationSettings, SpeedyGaussianRasterizer
from splatwizard.rasterizer.trim3dgs import  Trim3DGSRasterizationSettings, Trim3DGSGaussianRasterizer

from .config import SpeedySplatNormalModelParams, SpeedySplatNormalOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel
from ..._cmod.fused_ssim import fused_ssim
from ..._cmod.knn import knn_points
from ...metrics.loss_utils import l1_func, union_ssim_func
from ...scene import CameraIterator
from ...utils.sh_utils import eval_sh
from ...modules.dataclass import RenderResult, LossPack

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation,
)



@dataclass
class SSNRenderResult(RenderResult):
    extra_feats: torch.Tensor = None
    surf_depth: typing.Union[torch.Tensor, None] = None
    normal_reg_loss: typing.Union[torch.Tensor, None] = None
    step: int = None




def generate_grid(x_low, x_high, x_num, y_low, y_high, y_num, device):
    xs = torch.linspace(x_low, x_high, x_num, device=device)
    ys = torch.linspace(y_low, y_high, y_num, device=device)
    xv, yv = torch.meshgrid([xs, ys], indexing='xy')
    grid = torch.stack((xv.flatten(), yv.flatten())).T
    return grid


def compute_gradient(image, RGB2GRAY=False):
    assert image.ndim == 4, "image must have 4 dimensions"
    assert image.shape[1] == 1 or image.shape[1] == 3, "image must have 1 or 3 channels"
    if image.shape[1] == 3:
        assert RGB2GRAY == True, "RGB image must be converted to grayscale first"
        image = rgb_to_gray(image)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    image_for_pad = F.pad(image, pad=(1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(image_for_pad, sobel_kernel_x) / 3
    gradient_y = F.conv2d(image_for_pad, sobel_kernel_y) / 3

    return gradient_x, gradient_y


def rgb_to_gray(image):
    gray_image = (0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] +
                  0.114 * image[:, 2, :, :])
    gray_image = gray_image.unsqueeze(1)

    return gray_image


def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center




def normal_regularization(
        viewpoint_cam,
        gaussians: GaussianModel,
        pipe: PipelineParams,
        bg, visibility_filter, depth_grad_thresh=-1.0, close_thresh=1.0, dilation=2, depth_grad_mask_dilation=0):
    pos3D = gaussians.xyz
    pos3D = torch.cat((pos3D, torch.ones_like(pos3D[:, :1])), dim=1) @ viewpoint_cam.world_view_transform
    gs_camera_z = pos3D[:, 2:3]
    render_result: RenderResult =  gaussians.normal_render(viewpoint_cam, bg, pipe, override_color=gs_camera_z.repeat(1, 3)) #["render"][0]
    depth_map = render_result.rendered_image[0]
    if depth_grad_thresh > 0:
        depth_grad_x, depth_grad_y = compute_gradient(depth_map[None, None])
        depth_grad_mag = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2).squeeze()
        depth_grad_weight = (depth_grad_mag < depth_grad_thresh).float()
        if depth_grad_mask_dilation > 0:
            mask_di = depth_grad_mask_dilation
            depth_grad_weight = -1 * F.max_pool2d(-1 * depth_grad_weight[None, None, ...], mask_di * 2 + 1, stride=1, padding=mask_di).squeeze()

    grid = generate_grid(
        0.5 / depth_map.shape[-1], 1 - 0.5 / depth_map.shape[-1], depth_map.shape[-1],
        0.5 / depth_map.shape[-2], 1 - 0.5 / depth_map.shape[-2], depth_map.shape[-2], depth_map.device
    )
    depth = depth_map.view(-1, 1)
    # pixel to NDC
    pos = 2 * grid - 1
    # NDC to camera space
    pos[:, 0:1] = (pos[:, 0:1] - viewpoint_cam.projection_matrix[2, 0]) * depth / viewpoint_cam.projection_matrix[0, 0]
    pos[:, 1:2] = (pos[:, 1:2] - viewpoint_cam.projection_matrix[2, 1]) * depth / viewpoint_cam.projection_matrix[1, 1]
    pos_world = torch.cat((pos, depth, torch.ones_like(depth)), dim=-1) @ viewpoint_cam.world_view_transform.inverse()
    pos_world = pos_world[:, :3].permute(1, 0).view(1, 3, *depth_map.shape[-2:])
    pad_pos = F.pad(pos_world, (dilation, ) * 4, mode='replicate')
    di = dilation
    di2x = dilation * 2
    vec1 = pad_pos[:, :, di2x:, di2x:] - pad_pos[:, :, :-di2x, :-di2x]
    vec2 = pad_pos[:, :, :-di2x, di2x:] - pad_pos[:, :, di2x:, :-di2x]
    normal1 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    vec1 = pad_pos[:, :, di:-di, di2x:] - pad_pos[:, :, di:-di, :-di2x]
    vec2 = pad_pos[:, :, :-di2x, di:-di] - pad_pos[:, :, di2x:, di:-di]
    normal2 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    normal = F.normalize(normal1 + normal2, p=2, dim=0)
    dir_pp = (viewpoint_cam.camera_center.view(1, 3, 1, 1) - pos_world)
    dir_pp_normalized = F.normalize(dir_pp, p=2, dim=1).squeeze()
    normal = normal * torch.sign((normal * dir_pp_normalized).sum(0, keepdim=True))

    # normal_cam = (normal.flatten(1).T @ viewpoint_cam.world_view_transform[:3, :3]).T.view(3, *depth_map.shape[-2:])

    gs_normal = gaussians.normal
    dir_pp = (viewpoint_cam.camera_center.repeat(gaussians.features.shape[0], 1) - gaussians.xyz)
    dir_pp_normalized = F.normalize(dir_pp, p=2, dim=1)
    gs_normal = gs_normal * torch.sign((gs_normal * dir_pp_normalized).sum(1, keepdim=True))
    render_result: RenderResult = gaussians.normal_render(viewpoint_cam, bg, pipe, override_color=gs_normal) #["render"]
    pred_normal = render_result.rendered_image
    pred_normal = F.normalize(pred_normal, p=2, dim=0)
    # pred_normal_cam = (pred_normal.flatten(1).T @ viewpoint_cam.world_view_transform[:3, :3]).T.view(3, *depth_map.shape[-2:])

    pred_normal_shift = (pred_normal + 1) / 2
    normal_shift = (normal + 1) / 2

    if depth_grad_thresh > 0:
        normal_loss = (torch.abs(pred_normal_shift - normal_shift) * depth_grad_weight[None]).mean()
    else:
        normal_loss = torch.abs(pred_normal_shift - normal_shift).mean()

    valid_indices = torch.nonzero(visibility_filter).squeeze()
    pos2D = pos3D @ viewpoint_cam.projection_matrix
    ndc_coords = pos2D[:, :2] / pos2D[:, 3:4]
    gs_depthmap_z = F.grid_sample(depth_map[None, None], ndc_coords[valid_indices][None, None], align_corners=True).squeeze()
    close_mask = (gs_depthmap_z - gs_camera_z[valid_indices, 0]).abs() < close_thresh
    valid_indices = valid_indices[close_mask]

    closest_indices = gaussians.knn_idx[valid_indices]
    valid_normal = gs_normal[valid_indices]
    valid_shift_normal = (valid_normal + 1) / 2
    closest_normal = gs_normal[closest_indices]
    closest_normal = closest_normal * torch.sign((closest_normal * valid_normal[:, None]).sum(dim=-1, keepdim=True)).detach()
    closest_mean_shift_normal = (F.normalize(closest_normal.mean(1), p=2, dim=-1) + 1) / 2
    smooth_loss = torch.abs(valid_shift_normal - closest_mean_shift_normal).mean()
    return normal_loss + smooth_loss, normal


class SpeedySplatNormal(DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: SpeedySplatNormalModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.knn_to_track = None
        self.knn_dists = None
        self.knn_idx = None
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: SpeedySplatNormalOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

        scheduler.register_task(range(1, opt.iterations, opt.neighbor_reset_interval),
                                task=self.reset_neighbor, logging=True)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: SpeedySplatNormalOptimizationParams):
        # Densification
        # if iteration < opt.densify_until_iter:
        #     # Keep track of max radii in image-space for pruning
        #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
        #                                                          radii[visibility_filter])
        #     self.add_densification_stats(viewspace_point_tensor, visibility_filter)

        scheduler.register_task(range(opt.densify_until_iter), task=self.add_densification_stats)

        # if iteration < opt.densify_until_iter:
        #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        scheduler.register_task(
            range(opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval),
            task=self.densify_and_prune_task, logging=True
        )

        # --- Soft Pruning ---
        scheduler.register_task(
            range(
                opt.prune_from_iter,
                min(opt.densify_until_iter, opt.densify_until_iter),
                opt.prune_interval
            ),
            task=self.soft_prune_task, logging=True
        )

        # if iteration < opt.densify_until_iter:
        #     if iteration % opt.opacity_reset_interval == 0 or (
        #             dataset.white_background and iteration == opt.densify_from_iter):
        #         gaussians.reset_opacity()
        if ppl.white_background:
            scheduler.register_task(opt.densify_from_iter, task=self.reset_opacity, logging=True)
        scheduler.register_task(range(0, opt.densify_until_iter, opt.opacity_reset_interval),
                                task=self.reset_opacity, logging=True)


        # --- Hard Pruning ---
        scheduler.register_task(
            range(
                max(opt.densify_until_iter, opt.prune_from_iter),
                opt.prune_until_iter,
                opt.prune_interval
            ),
            task=self.hard_prune_task, logging=True
        )


    @property
    def normal(self):
        R = build_rotation(self.rotation)
        min_scale_axis = F.one_hot(torch.argmin(self.scaling, dim=-1), num_classes=3).float()
        gs_normal = torch.bmm(R, min_scale_axis[:, :, None]).squeeze(-1)
        gs_normal = F.normalize(gs_normal, dim=1)
        return gs_normal

    # @property
    # def normal(self):
    #     R = build_rotation(self.rotation)
    #     gs_normal = R[..., 0]
    #     gs_normal = F.normalize(gs_normal, dim=1)
    #     return gs_normal
    #
    #
    # @property
    # def scaling(self):
    #     self.s0 = torch.ones_like(self._scaling[:, :1]) * self.eps_s0
    #     return torch.cat((self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])), dim=1)

    @task
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
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

    @task
    def update_learning_rate(self, iteration: int):
        """
        Learning rate scheduling per step
        Args:
            iteration:

        Returns:

        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    @task
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # @task
    # def train_statis_task(self, render_result: RenderResult):
    #     self.add_densification_stats(render_result)

    @task
    def densify_and_prune_task(self, opt: SpeedySplatNormalOptimizationParams , step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

        self.raw_reset_neighbor(opt)

    def speedy_render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams = None, step=0, scaling_modifier=1.0, override_color=None,
               scores=None):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = SpeedyGaussianRasterizationSettings(
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

        rasterizer = SpeedyGaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        # set scores to the correct size if not passed in
        if scores is None:
            scores = torch.zeros_like(opacity)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
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

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, kernel_times = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scores=scores,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return SSNRenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )


    def normal_render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams=None,
               step=0, scaling_modifier=1.0, override_color=None, record_transmittance=False):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype,
            requires_grad=True, device="cuda") + 0
        if self._training:
            try:
                screenspace_points.retain_grad()
            except:
                pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = Trim3DGSRasterizationSettings(
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
            record_transmittance=record_transmittance,
            debug=pipe.debug
        )

        rasterizer = Trim3DGSGaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
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

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # color, out_extra_feats, median_depth, radii
        output = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
            return transmittance
        else:
            rendered_image, rendered_extra_feats, median_depth, radii = output

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter": radii > 0,
        #         "radii": radii}
        # if opt is not None and opt.use_trained_exposure:
        #     exposure = self.get_exposure_from_name(viewpoint_camera.image_name)
        #     rendered_image = torch.matmul(
        #         rendered_image.permute(1, 2, 0),
        #         exposure[:3, :3]
        #     ).permute(2, 0, 1) + exposure[:3, 3, None, None]

        # invdepths = -invdepths
        # invdepths = invdepths - invdepths.min()
        return SSNRenderResult(
            rendered_image=rendered_image,
            extra_feats=rendered_extra_feats,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii,
            surf_depth=median_depth
        )


    def render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: SpeedySplatNormalOptimizationParams = None, step=0, scaling_modifier=1.0, override_color=None,
               scores=None, record_transmittance=False):

        if opt is None and scores is None: # eval mode
            return self.normal_render(viewpoint_camera, bg_color,
               pipe, opt, step, scaling_modifier, override_color, record_transmittance=record_transmittance)
        if step < opt.normal_regularity_from_iter or step > opt.normal_regularity_until_iter:
            return self.speedy_render(viewpoint_camera, bg_color,
               pipe, opt, step, scaling_modifier, override_color, scores)

        if scores is not None:
            return self.speedy_render(viewpoint_camera, bg_color,
                                          pipe, opt, step, scaling_modifier, override_color, scores)

        render_result = self.normal_render(viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color, record_transmittance)
        # print('normal render result:')
        normal_reg_loss, normal = normal_regularization(
            viewpoint_camera, self, pipe, torch.zeros_like(bg_color), render_result.visibility_filter,
            depth_grad_thresh=opt.depth_grad_thresh,
            close_thresh=opt.normal_close_thresh,
            dilation=opt.normal_dilation,
            depth_grad_mask_dilation=opt.depth_grad_mask_dilation)
        if torch.isnan(normal_reg_loss).any():
            logger.info('Got NaN in normal loss, skip')
            normal_reg_loss = 0

        render_result.normal_reg_loss = normal_reg_loss
        render_result.step = step
        return render_result

    def loss_func(self, viewpoint_cam, render_result: SSNRenderResult, opt: SpeedySplatNormalOptimizationParams) -> (torch.Tensor,
                                                                                                     LossPack):
            gt_image = viewpoint_cam.original_image
            Ll1 = l1_func(render_result.rendered_image, gt_image)

            ssim_value = union_ssim_func(render_result.rendered_image, gt_image, using_fused=opt.use_fused_ssim)


            ssim_loss = (1.0 - ssim_value)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

            if render_result.step and  render_result.step >= opt.normal_regularity_from_iter and render_result.step < opt.normal_regularity_until_iter:
                normal_reg_loss = render_result.normal_reg_loss
                # print(render_result.step)
                # # scale regularizatoin
                # scale_reg_loss = scale_regularization(gaussians, iteration, opt)
                # loss = loss + scale_reg_loss
                loss += opt.normal_regularity_param * normal_reg_loss
            loss_pack = LossPack(
                l1_loss=Ll1,
                ssim_loss=ssim_loss,
                loss=loss
            )

            return loss, loss_pack

    # def normal_loss_func(self, viewpoint_cam, render_result: SSNRenderResult, opt: SpeedySplatNormalOptimizationParams) -> (torch.Tensor,
    #                                                                                                  LossPack):
    #     # Loss
    #     gt_image = viewpoint_cam.original_image.cuda()
    #
    #     Ll1 = l1_func(render_result.rendered_image, gt_image)
    #
    #     ssim_value = union_ssim_func(render_result.rendered_image, gt_image, using_fused=opt.use_fused_ssim)
    #
    #     ssim_loss = (1.0 - ssim_value)
    #
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
    #
    #     normal_reg_loss = render_result.norm_reg_loss
    #     # # scale regularizatoin
    #     # scale_reg_loss = scale_regularization(gaussians, iteration, opt)
    #     # loss = loss + scale_reg_loss
    #     loss += opt.normal_regularity_param * normal_reg_loss
    #     loss_pack = LossPack(
    #         l1_loss=Ll1,
    #         ssim_loss=ssim_loss,
    #         loss=loss
    #     )
    #
    #     return loss, loss_pack


    def score_func(self, view, pipeline, background, scores):

        img_scores = torch.zeros_like(scores)
        img_scores.requires_grad = True

        image = self.speedy_render(view, background, pipeline,
                       scores=img_scores).rendered_image

        # Backward computes and stores grad squared values
        # in img_scores's grad
        image.sum().backward()

        scores += img_scores.grad

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1

    def prune(self, cam_iter, pipe, background, prune_ratio):

        # iter_start = torch.cuda.Event(enable_timing=True)
        # iter_end = torch.cuda.Event(enable_timing=True)
        # torch.cuda.reset_peak_memory_stats()

        # iter_start.record()

        with torch.enable_grad():
            pbar = tqdm(
                total=len(cam_iter),
                desc='Computing Pruning Scores')
            scores = torch.zeros_like(self.opacity)
            for view in cam_iter:
                self.score_func(view, pipe, background,
                           scores)
                pbar.update(1)
            pbar.close()

        self.prune_gaussians(prune_ratio, scores)

        # iter_end.record()
        #
        # # Track peak memory usage (in bytes) and convert to MB
        # peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        # time_ms = iter_start.elapsed_time(iter_end)
        # time_min = time_ms / 60_000

        # return {
        #     "peak_memory_allocated": peak_memory_allocated,
        #     "peak_memory_reserved": peak_memory_reserved,
        #     "time_min": time_min
        # }

    @task
    def soft_prune_task(self, ppl: PipelineParams, opt: SpeedySplatNormalOptimizationParams, cam_iterator: CameraIterator):
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        # prune_pkg =
        self.prune(cam_iterator, ppl, background, opt.densify_prune_ratio)

        # prune_time_min += prune_pkg['time_min']
        # prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
        # prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']

        self.raw_reset_neighbor(opt)

    @task
    def hard_prune_task(self, ppl: PipelineParams, opt: SpeedySplatNormalOptimizationParams, cam_iterator: CameraIterator):
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        # prune_pkg =
        self.prune(cam_iterator, ppl, background, opt.after_densify_prune_ratio)

        # prune_time_min += prune_pkg['time_min']
        # prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
        # prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']
        self.raw_reset_neighbor(opt)

    @task
    def reset_neighbor(self, opt: SpeedySplatNormalOptimizationParams):
        self.raw_reset_neighbor(opt)

    @torch.no_grad()
    def raw_reset_neighbor(self, opt: SpeedySplatNormalOptimizationParams):
            self.knn_to_track = opt.knn_to_track
            knns = knn_points(self.xyz[None], self.xyz[None], K=opt.knn_to_track, return_sorted=False)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]
