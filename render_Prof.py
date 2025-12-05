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
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        print("[INFO] Rendering view {}".format(idx))
        render_pkg = render(view, gaussians, pipeline, background, obb_flag=True)
        rendering = render_pkg["render"]
        buffer = render_pkg["buffer"]

        # generate sim input
        if idx == 60 or idx == 61:
            sim_input_path_buffer = os.path.join(model_path, "prof_raw_data/iteration_{}".format(iteration), "frame_{}_buffer_raw_data.pt".format(idx))           
            mkdir_p(os.path.dirname(sim_input_path_buffer))
            torch.save(buffer, sim_input_path_buffer)

        # static fragments
        binningBuffer = buffer["binningBuffer"]
        list_num = buffer["num_rendered"]
        bin_dict = get_bin_buffer_offset(list_num)
        point_list_buffer = binningBuffer[bin_dict["point_list"]:bin_dict["point_list"]+4*list_num]
        point_list_data = pack_byte_data(point_list_buffer, np.uint32)
        print("[INFO] Fragment Number: {}".format(len(point_list_data)))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


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
