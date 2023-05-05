import os
import argparse
import collections
import glob
import pickle
from loguru import logger
from typing import List, Dict, Set
import numpy as np
import torch
from common.colmap import read_write_model
import common.math_utils as math_utils
from PIL import Image, ImageOps

DataPair = collections.namedtuple(
    "DataPair",
    ["camera_path_1", "intrinsics_1",
     "camera_path_2", "intrinsics_2",
     "rotation_1_to_2", "translation_1_to_2", "fundamental_1_to_2"])

# example : https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class RelativePoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, pickle_path, resize_ratio=0.5):
        self.data_folder = data_folder
        file = open(pickle_path, 'rb')
        self.all_data_pairs = pickle.load(file)
        logger.info("load %d pairs" % len(self.all_data_pairs))
        self.resize_ratio = float(resize_ratio)

    def __len__(self):
        return len(self.all_data_pairs)

    def get_image(self, image_path):
        image_gray = ImageOps.grayscale(Image.open(image_path))
        width, height = image_gray.size
        new_width = int(float(width) * self.resize_ratio)
        new_height = int(float(height) * self.resize_ratio)
        image_gray = image_gray.resize((new_width, new_height))
        image = np.asarray(image_gray)
        return np.expand_dims(image, axis=0)

    def __getitem__(self, idx):
        # load images
        data_pair = self.all_data_pairs[idx]

        image_1 = self.get_image(os.path.join(self.data_folder, data_pair.camera_path_1))
        image_2 = self.get_image(os.path.join(self.data_folder, data_pair.camera_path_2))

        # add a factor to intrinsics to make them close to 1
        factor_param = self.resize_ratio / 1000.0
        param_1 = factor_param * math_utils.intrinsics_matrix_to_param(data_pair.intrinsics_1);
        param_2 = factor_param * math_utils.intrinsics_matrix_to_param(data_pair.intrinsics_2);

        # get relative pose array: rotation to quaternion, translation will be normalized
        # rescale qvec by factor of 10
        qvec = 10 * read_write_model.rotmat2qvec(data_pair.rotation_1_to_2);
        trans_norm = np.linalg.norm(data_pair.translation_1_to_2)
        tvec = data_pair.translation_1_to_2 / trans_norm
        target_pose = np.array([tvec[0], tvec[1], tvec[2], qvec[0], qvec[1], qvec[2], qvec[3]])

        return {
            "image_1": image_1,
            "param_1": param_1.astype(np.float32),
            "image_2": image_2,
            "param_2": param_2.astype(np.float32),
            "target_pose": target_pose.astype(np.float32),
        }


def get_points3d_overlap(points1: Set, points2: Set) -> float:
    return len(points1 & points2) / min(len(points1), len(points2))


def get_relative_fundamental_matrix(images, cameras, image_id1, image_id2):
    # get poses for each image
    image_1 = images[image_id1]
    image_2 = images[image_id2]
    world_to_camera_1_rot = read_write_model.qvec2rotmat(image_1.qvec)
    world_to_camera_1_trans = image_1.tvec
    world_to_camera_2_rot = read_write_model.qvec2rotmat(image_2.qvec)
    world_to_camera_2_trans = image_2.tvec

    rotation_1_to_2 = world_to_camera_2_rot.dot(np.transpose(world_to_camera_1_rot))
    translation_1_to_2 = world_to_camera_2_trans - rotation_1_to_2.dot(world_to_camera_1_trans)

    essential_1_to_2 = math_utils.relative_pose_to_essential_matrix(rotation_1_to_2, translation_1_to_2)
    intri_matrix_1_inv = read_write_model.get_inverse_intrinsics_matrix(cameras[image_1.camera_id])
    intri_matrix_2_inv = read_write_model.get_inverse_intrinsics_matrix(cameras[image_2.camera_id])
    return ((np.transpose(intri_matrix_2_inv)).dot(essential_1_to_2)).dot(intri_matrix_1_inv)


def get_relative_pose_and_intrinsics(images, cameras, image_id1, image_id2):
    image_1 = images[image_id1]
    image_2 = images[image_id2]
    world_to_camera_1_rot = read_write_model.qvec2rotmat(image_1.qvec)
    world_to_camera_1_trans = image_1.tvec
    world_to_camera_2_rot = read_write_model.qvec2rotmat(image_2.qvec)
    world_to_camera_2_trans = image_2.tvec

    rotation_1_to_2 = world_to_camera_2_rot.dot(np.transpose(world_to_camera_1_rot))
    translation_1_to_2 = world_to_camera_2_trans - rotation_1_to_2.dot(world_to_camera_1_trans)
    intri_matrix_1 = read_write_model.get_intrinsics_matrix(cameras[image_1.camera_id])
    intri_matrix_2 = read_write_model.get_intrinsics_matrix(cameras[image_2.camera_id])

    return intri_matrix_1, intri_matrix_2, rotation_1_to_2, translation_1_to_2


def prepare_pairs(session_folder,
                  model_folder = "/colmap/sparse/",
                  images_folder = "/colmap/images/",
                  overlap_interval = [0.1, 0.7]):
    # create image pairs from colmap model

    # load colmap model
    model_path = session_folder + model_folder
    cameras = read_write_model.read_cameras(model_path + "cameras");
    images_raw = read_write_model.read_images(model_path + "images");
    # points3d = read_write_model.read_points3d(model_path + "points3D");

    # make images points to set
    images = {}
    for idx, (image_id, image_info1) in enumerate(images_raw.items()):
        images[image_id] = set(image_info1.point3D_ids.tolist())

    # create pairs based on shared observations
    image_pairs = []
    for idx, (image_id, point_ids1) in enumerate(images.items()):
        if idx % 1000 == 0:
            logger.info(f"{idx}/{len(images)}")
        for idx2, (image_id2, point_ids2) in enumerate(images.items()):
            if image_id >= image_id2:
                continue
            overlap = get_points3d_overlap(point_ids1, point_ids2)
            if overlap_interval[0] <= overlap <= overlap_interval[1]:
                image_pairs.append([image_id, image_id2]) # save the pair

    logger.info("  Find " + str(len(image_pairs)) + " pairs")
    logger.info("  Compute fundamental matrices")

    session_name = session_folder.split('/')[-1]
    data_pairs = []
    for i in range(len(image_pairs)):
        # images_folder
        image_path_1 = session_name + images_folder + images_raw[image_pairs[i][0]].name
        image_path_2 = session_name + images_folder + images_raw[image_pairs[i][1]].name

        fundamental_1_to_2 = get_relative_fundamental_matrix(
            images_raw, cameras, image_pairs[i][0], image_pairs[i][1])
        intri_matrix_1, intri_matrix_2, rotation_1_to_2, translation_1_to_2 = get_relative_pose_and_intrinsics(
            images_raw, cameras, image_pairs[i][0], image_pairs[i][1])
        data_pairs.append(DataPair(
            camera_path_1 = image_path_1,
            intrinsics_1 = intri_matrix_1,
            camera_path_2 = image_path_2,
            intrinsics_2 = intri_matrix_2,
            rotation_1_to_2 = rotation_1_to_2,
            translation_1_to_2 = translation_1_to_2,
            fundamental_1_to_2 = fundamental_1_to_2))

    return data_pairs



# python relative_pose/dataset.py /mnt/gz01/experiment/liuye/relative_pose /mnt/gz01/experiment/liuye/relative_pose
def main(args):
    # query all the sessions in the input folder
    session_folders = glob.glob(args.sessions_input_folder + "/20*")
    all_data_pairs = []
    for session_folder in session_folders:
        logger.info("process " + session_folder)
        data_pairs = prepare_pairs(session_folder)
        all_data_pairs += data_pairs

    # save result to file
    logger.info("Save to " + args.output_folder)
    if not os.path.exists(args.output_folder):
       os.makedirs(args.output_folder)

    # open a file, where you ant to store the data
    file = open(os.path.join(args.output_folder, "all_data_pairs.pickle"), 'wb')
    # dump information to that file
    pickle.dump(all_data_pairs, file)

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read colmap models and output pairs')
    parser.add_argument('sessions_input_folder', help='path to input model folder')
    parser.add_argument('output_folder', help='path to output folder')
    args = parser.parse_args()
    main(args)
