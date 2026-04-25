# import sys
#
# from splatwizard.modules.mesh import cull_scan
#
# import numpy as np
# import open3d as o3d
# import sklearn.neighbors as skln
# from tqdm import tqdm
# from scipy.io import loadmat
# import multiprocessing as mp
# import argparse
#
# from splatwizard.pipeline.reconstruct_model import reconstruct_model
#
#
# def sample_single_tri(input_):
#     n1, n2, v1, v2, tri_vert = input_
#     c = np.mgrid[:n1 + 1, :n2 + 1]
#     c += 0.5
#     c[0] /= max(n1, 1e-7)
#     c[1] /= max(n2, 1e-7)
#     c = np.transpose(c, (1, 2, 0))
#     k = c[c.sum(axis=-1) < 1]  # m2
#     q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
#     return q
#
#
# def write_vis_pcd(file, points, colors):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_point_cloud(file, pcd)
#
#
# def evaluate():
#     mp.freeze_support()
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default='data_in.ply')
#     parser.add_argument('--scan', type=int, default=1)
#     parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
#     parser.add_argument('--dataset_dir', type=str, default='.')
#     parser.add_argument('--vis_out_dir', type=str, default='.')
#     parser.add_argument('--downsample_density', type=float, default=0.2)
#     parser.add_argument('--patch_size', type=float, default=60)
#     parser.add_argument('--max_dist', type=float, default=20)
#     parser.add_argument('--visualize_threshold', type=float, default=10)
#     args = parser.parse_args()
#
#     thresh = args.downsample_density
#     if args.mode == 'mesh':
#         pbar = tqdm(total=9)
#         pbar.set_description('read data mesh')
#         data_mesh = o3d.io.read_triangle_mesh(args.data)
#
#         vertices = np.asarray(data_mesh.vertices)
#         triangles = np.asarray(data_mesh.triangles)
#         tri_vert = vertices[triangles]
#
#         pbar.update(1)
#         pbar.set_description('sample pcd from mesh')
#         v1 = tri_vert[:, 1] - tri_vert[:, 0]
#         v2 = tri_vert[:, 2] - tri_vert[:, 0]
#         l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
#         l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
#         area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
#         non_zero_area = (area2 > 0)[:, 0]
#         l1, l2, area2, v1, v2, tri_vert = [
#             arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
#         ]
#         thr = thresh * np.sqrt(l1 * l2 / area2)
#         n1 = np.floor(l1 / thr)
#         n2 = np.floor(l2 / thr)
#
#         with mp.Pool() as mp_pool:
#             new_pts = mp_pool.map(sample_single_tri,
#                                   ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
#                                    range(len(n1))), chunksize=1024)
#
#         new_pts = np.concatenate(new_pts, axis=0)
#         data_pcd = np.concatenate([vertices, new_pts], axis=0)
#
#     elif args.mode == 'pcd':
#         pbar = tqdm(total=8)
#         pbar.set_description('read data pcd')
#         data_pcd_o3d = o3d.io.read_point_cloud(args.data)
#         data_pcd = np.asarray(data_pcd_o3d.points)
#
#     pbar.update(1)
#     pbar.set_description('random shuffle pcd index')
#     shuffle_rng = np.random.default_rng()
#     shuffle_rng.shuffle(data_pcd, axis=0)
#
#     pbar.update(1)
#     pbar.set_description('downsample pcd')
#     nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
#     nn_engine.fit(data_pcd)
#     rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
#     mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
#     for curr, idxs in enumerate(rnn_idxs):
#         if mask[curr]:
#             mask[idxs] = 0
#             mask[curr] = 1
#     data_down = data_pcd[mask]
#
#     pbar.update(1)
#     pbar.set_description('masking data pcd')
#     obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
#     ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
#     BB = BB.astype(np.float32)
#
#     patch = args.patch_size
#     inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
#     data_in = data_down[inbound]
#
#     data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
#     grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
#     data_grid_in = data_grid[grid_inbound]
#     in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
#     data_in_obs = data_in[grid_inbound][in_obs]
#
#     pbar.update(1)
#     pbar.set_description('read STL pcd')
#     stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
#     stl = np.asarray(stl_pcd.points)
#
#     pbar.update(1)
#     pbar.set_description('compute data2stl')
#     nn_engine.fit(stl)
#     dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
#     max_dist = args.max_dist
#     mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
#
#     pbar.update(1)
#     pbar.set_description('compute stl2data')
#     ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']
#
#     stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
#     above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
#     stl_above = stl[above]
#
#     nn_engine.fit(data_in)
#     dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
#     mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
#
#     pbar.update(1)
#     pbar.set_description('visualize error')
#     vis_dist = args.visualize_threshold
#     R = np.array([[1, 0, 0]], dtype=np.float64)
#     G = np.array([[0, 1, 0]], dtype=np.float64)
#     B = np.array([[0, 0, 1]], dtype=np.float64)
#     W = np.array([[1, 1, 1]], dtype=np.float64)
#     data_color = np.tile(B, (data_down.shape[0], 1))
#     data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
#     data_color[np.where(inbound)[0][grid_inbound][in_obs]] = R * data_alpha + W * (1 - data_alpha)
#     data_color[np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:, 0] >= max_dist]] = G
#     write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
#     stl_color = np.tile(B, (stl.shape[0], 1))
#     stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
#     stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
#     stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
#     write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)
#
#     pbar.update(1)
#     pbar.set_description('done')
#     pbar.close()
#     over_all = (mean_d2s + mean_s2d) / 2
#     print(mean_d2s, mean_s2d, over_all)
#
#     import json
#
#     with open(f'{args.vis_out_dir}/results.json', 'w') as fp:
#         json.dump({
#             'mean_d2s': mean_d2s,
#             'mean_s2d': mean_s2d,
#             'overall': over_all,
#         }, fp, indent=True)
#
#
#
# # def main():
#     # parser.add_argument('--input_mesh', type=str, help='path to the mesh to be evaluated')
#     # parser.add_argument('--scan_id', type=str, help='scan id of the input mesh')
#     # parser.add_argument('--output_dir', type=str, default='evaluation_results_single',
#     #                     help='path to the output folder')
#     # parser.add_argument('--mask_dir', type=str, default='mask', help='path to uncropped mask')
#     # parser.add_argument('--DTU', type=str, default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')
#     # args = parser.parse_args()
#     #
#     # Offical_DTU_Dataset = args.DTU
#     # out_dir = args.output_dir
#     # Path(out_dir).mkdir(parents=True, exist_ok=True)
#
#     # scan = args.scan_id
#     # ply_file = args.input_mesh
#     # print("cull mesh ....")
#     # result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
#     # cull_scan(
#     #     '/tmp_output/liuxiang/output/splatwizard/surfgs_dev/point_cloud/iteration_30000/point_cloud.ply',
#     #     '/tmp_output/liuxiang/output/splatwizard/surfgs_dev/mesh_file',
#     #     instance_dir='/tmp_output/liuxiang/dataset/DTU/scan24/'
#     # )
#
#     # script_dir = os.path.dirname(os.path.abspath(__file__))
#     # cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
#     # os.system(cmd)
#     #
#     # print("export mesh ...")
#     #
#     # # set the active_sh to 0 to export only diffuse texture
#     # gaussExtractor.gaussians.active_sh_degree = 0
#     # gaussExtractor.reconstruction(scene.getTrainCameras())
#     # # extract the mesh and save
#     # if args.unbounded:
#     #     name = 'fuse_unbounded.ply'
#     #     mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
#     # else:
#     #     name = 'fuse.ply'
#     #     depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
#     #     voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
#     #     sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
#     #     mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
#     #
#     # o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
#     # print("mesh saved at {}".format(os.path.join(train_dir, name)))
#     # # post-process the mesh and save, saving the largest N clusters
#     # mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
#     # o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
#     # print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

import pathlib
import sys
from enum import Enum

from simple_parsing import ArgumentParser
from loguru import logger


from splatwizard.modules.dataclass import TrainContext

from splatwizard.config import PipelineParams, OptimizationParams, EvalMode, MeshExtractorParams
from splatwizard.model_zoo import CONFIG_CACHE
from splatwizard.pipeline.reconstruct_model import reconstruct_model

from splatwizard.utils.logging import setup_tensorboard
from splatwizard.utils.misc import safe_state


def setup_output_dir(pp: PipelineParams, train_context: TrainContext, stage: Enum=None):
    # 创建输出目录

    if pp.output_dir is None:
        return
    if stage is not None:
        train_context.output_dir = train_context.base_output_dir / str(stage.name)
        train_context.output_dir.mkdir(exist_ok=True)
    else:
        train_context.output_dir = train_context.base_output_dir

    train_context.checkpoint_dir = train_context.output_dir / 'checkpoints'
    train_context.checkpoint_dir.mkdir(exist_ok=True)

    # if pp.env_path is not None:
    #     os.environ["PATH"] = pp.env_path + ':' + os.environ["PATH"]

    # 初始化日志文件
    logger.add(train_context.output_dir / 'recon.log')
    # logger.info(f'running tag: {args.tag}')

    if pp.dataset is None:
        dataset = pathlib.Path(pp.source_path).name
    else:
        dataset = pp.dataset

    tb_writer = setup_tensorboard(train_context.output_dir)
    tb_writer.prefix = dataset

    train_context.tb_writer = tb_writer

    logger.info("Setting up output dir" + str(train_context.output_dir.absolute()))

    return train_context


def validate_pipeline_parameters(pp: PipelineParams):
    if pp.lanczos_resample:
        assert pp.num_workers == 0, "Only single worker mode supports lanczos_resample=True"

    assert pp.checkpoint_type in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'

    if isinstance(pp.final_checkpoint, str):
        assert pp.final_checkpoint in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'
    else:
        for type_ in pp.final_checkpoint:
            assert type_ in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(PipelineParams, dest="pipeline")  # noqa
    parser.add_arguments(CONFIG_CACHE[0], dest="model_group")  # noqa
    parser.add_arguments(MeshExtractorParams, dest="mesh_group") # noqa

    args = parser.parse_args(sys.argv[1:])

    mp = args.model_group.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.model_group.optim
    mep: MeshExtractorParams = args.mesh_group

    validate_pipeline_parameters(pp)

    if pp.seed is not None:
        safe_state(pp.seed)
    train_context = TrainContext()
    train_context.model = args.subgroups['model_group.model']

    if pp.output_dir is not None:
        # 创建输出目录
        train_context.base_output_dir = pathlib.Path(pp.output_dir)
        train_context.base_output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Output dir: " + str(train_context.base_output_dir.absolute()))

    if pp.eval_mode is None:
        pp.eval_mode = EvalMode.NORMAL

    logger.info(f"{pp}")
    logger.info(f"{mp}")
    logger.info(f"{op}")

    setup_output_dir(pp, train_context)

    reconstruct_model(pp, mp, op, mep,  train_context)

if __name__ == '__main__':
    sys.exit(main())
