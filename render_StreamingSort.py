#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.util_system import mkdir_p
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy

def insert_interpolated_poses(views, N: int):
    if N < 0:
        raise ValueError("N must be positive")
    elif N == 0:
        return views
    out = []
    for i in range(len(views) - 1):
        left = views[i]
        right = views[i + 1]
        out.append(left)

        left_pose = left.world_view_transform.cpu()
        right_pose = right.world_view_transform.cpu()

        # 提取平移向量 t (第四列的前三行)
        t_left = left_pose[:3, 3]
        t_right = right_pose[:3, 3]
        
        # 提取旋转矩阵 R (左上 3x3) 并转换为四元数 q
        # R.from_matrix 需要一个 3x3 的旋转矩阵
        q_left = R.from_matrix(left_pose[:3, :3]).as_quat()
        q_right = R.from_matrix(right_pose[:3, :3]).as_quat()
        endpoints = R.from_quat(np.array([q_left, q_right]))
        slerp_interpolator = Slerp([0, 1], endpoints)
        for j in range(1, N + 1):
            alpha = j / (N + 1)
            
            # --- 3. 旋转插值 (SLERP) ---
            # 必须在四元数空间进行球面线性插值 (SLERP) 来保证平滑的旋转
            r_interp = slerp_interpolator(alpha)
                        
            # --- 4. 平移插值 (LERP) ---
            t_interp = (1.0 - alpha) * t_left + alpha * t_right

            T_interp = np.eye(4)
            T_interp[:3, :3] = r_interp.as_matrix()    # 插入旋转矩阵
            T_interp[:3, 3] = t_interp     # 插入平移向量
            
            new_view = copy.deepcopy(left)
            # change to GPU float
            new_view.world_view_transform = torch.from_numpy(T_interp).float().to(left.world_view_transform.device)
            # align params that related to world_view_transform matrix
            new_view.full_proj_transform = (
            new_view.world_view_transform.unsqueeze(0).bmm(
                new_view.projection_matrix.unsqueeze(0)
            )
            ).squeeze(0)
            new_view.camera_center = new_view.world_view_transform.inverse()[3, :3]
            
            out.append(new_view)
    out.append(views[-1])
    return out

def pack_byte_data(buffer_in, dtype):
    '''
    Convert little-end raw byte data to target data type
    :param buffer_in: raw byte data (torch.Tensor or np.ndarray)
    :param dtype: target data type, only numpy dtype
    :return: The packed data in target data type
    '''
    if isinstance(buffer_in, torch.Tensor):
        buffer_in = np.array(buffer_in.to('cpu')).tobytes()
    packed_data = np.frombuffer(buffer_in, dtype=dtype)
    return packed_data

def get_bin_buffer_offset(P:int, align=128):
    '''
    Get the byte offset of target parameters in Binning Buffer(The duplicated key-value list after rasterization)
        :param P: the number of the duplicated Gaussian Points
        :param align: The boundary for byte alignment
        :return: The offset dict to target parameters
    '''
    align = align - 1
    target_dict = {}
    offset = 0
    target_dict["point_list"] = offset
    offset = (offset + P * 4 + align) & ~align  # unit32 point_list (GaussID)
    offset = (offset + P * 4 + align) & ~align  # unit32 point_list_unsorted (GaussID)
    target_dict["key_list"] = offset
    offset = (offset + P * 8 + align) & ~align  # uint64 point_list_keys (tile | depth)
    offset = (offset + P * 8 + align) & ~align  # uint64 point_list_keys_unsorted (tile | depth)
    return target_dict


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # It seems that interpolating view pose can not contribute to Sorting Error Elimination
    Interp_N = 0
    views = insert_interpolated_poses(views, Interp_N)
    print("[INFO] InterpEach Number: {}, Interpolated views number: {}".format(Interp_N, len(views)))

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # debug
        # if idx == 1:
        #     break
            
        print("[INFO] Rendering view {}".format(idx))
        # precisely sort once every k frames
        k = 10
        if idx % k == 0: # if idx == 0:
            prev_depth = None
        else:
            prev_depth = current_depth
        
        render_pkg = render(view, gaussians, pipeline, background, obb_flag=True, prev_depth=prev_depth)    
        
        current_depth = render_pkg["current_depth"]
        
        if idx%(Interp_N+1) == 0:
            # static fragments
            rendering = render_pkg["render"]
            buffer = render_pkg["buffer"]
            binningBuffer = buffer["binningBuffer"]
            list_num = buffer["num_rendered"]
            bin_dict = get_bin_buffer_offset(list_num)
            point_list_buffer = binningBuffer[bin_dict["point_list"]:bin_dict["point_list"]+4*list_num]
            point_list_data = pack_byte_data(point_list_buffer, np.uint32)
            print("[INFO] Fragment Number: {}".format(len(point_list_data)))
            # Save rendered images results
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, "{0:05d}".format(idx//(Interp_N+1)) + ".png")
            )
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, "{0:05d}".format(idx//(Interp_N+1)) + ".png")
            )

        # generate sim input
        if idx == 60*(Interp_N+1) or idx == 61*(Interp_N+1):
            sim_input_path_buffer = os.path.join(model_path, "prof_raw_data_streaming_sort/iteration_{}".format(iteration), "frame_{}_buffer_raw_data.pt".format(idx//(Interp_N+1)))           
            mkdir_p(os.path.dirname(sim_input_path_buffer))
            torch.save(buffer, sim_input_path_buffer)


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    load_vq: bool, 
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_vq= load_vq)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.load_vq
    )
