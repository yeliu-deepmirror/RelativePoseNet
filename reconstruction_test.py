import argparse
import glob
import numpy as np
from relative_pose.model.relate_pose_net import RelativePoseNet
import os
from loguru import logger
from torch.utils.data import DataLoader
from reconstruction.rotation_averaging import ShonanRotationAveraging
from reconstruction.translation_averaging import TranslationAveraging1DSFM, plot_camera
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import gtsam


def get_image(image_path, resize_ratio):
    image_gray = ImageOps.grayscale(Image.open(image_path))
    width, height = image_gray.size
    new_width = int(float(width) * resize_ratio)
    new_height = int(float(height) * resize_ratio)
    image_gray = image_gray.resize((new_width, new_height))
    image = np.asarray(image_gray)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(np.expand_dims(image, axis=0)).cuda()


def output_to_measure(output_np):
    translation = gtsam.Unit3(gtsam.Point3(-output_np[0], -output_np[1], -output_np[2]))
    rotation = gtsam.Rot3.Quaternion(
        0.1 * output_np[3], 0.1 * output_np[4], 0.1 * output_np[5], 0.1 * output_np[6])
    return translation, rotation


# python reconstruction/reconstruction_test.py
def reconstruction_test(args):
    # read images from folder
    image_paths = glob.glob(args.images_folder + "/*.jpg")
    logger.info("Find %d images" % len(image_paths))

    # load model
    model = torch.nn.DataParallel(RelativePoseNet())
    # model = RelativePoseNet()
    logger.info("load model from : %s" % args.model_path)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.cuda()
    model.eval()

    # process sequence match with the images
    num_match_neighbor = 5
    resize_ratio = 0.25
    matches = {}

    factor_param = resize_ratio / 1000.0
    param_intri = factor_param * np.array([876.962, 883.628, 933.231, 601.85]).astype(np.float32)
    param_intri = np.expand_dims(param_intri, axis=0)
    param_intri = torch.from_numpy(param_intri).cuda()
    for id_1 in range(len(image_paths)):
        for k in range(num_match_neighbor):
            id_2 = id_1 + k + 1
            if id_2 >= len(image_paths):
                continue
            # match the two images
            image_1 = get_image(image_paths[id_1], resize_ratio)
            image_2 = get_image(image_paths[id_2], resize_ratio)
            output = model(image_1, param_intri, image_2, param_intri).cpu().detach().numpy()
            matches[(id_1, id_2)] = output[0]
            # logger.info(f"{id_1} {id_2} {output[0]}")

            # clean cache
            del image_1
            del image_2
            torch.cuda.empty_cache()

        if id_1%10 == 0:
            logger.info(f"  Process {id_1} images, {len(matches)} matches.")

    del model
    del param_intri
    torch.cuda.empty_cache()

    # run reconstruction
    rotation_averaging = ShonanRotationAveraging()
    rotation_i1_to_i2_dict = {}
    translation_i1_to_i2_dict = {}
    for idx, ((i1, i2), ouput) in enumerate(matches.items()):
        translation, rotation = output_to_measure(ouput)
        rotation_i1_to_i2_dict[(i1, i2)] = rotation
        translation_i1_to_i2_dict[(i1, i2)] = translation

    logger.info("Run ShonanRotationAveraging")
    rotation_averager = ShonanRotationAveraging()
    rotations = rotation_averager.run_rotation_averaging(len(image_paths), rotation_i1_to_i2_dict)

    logger.info("Run TranslationAveraging1DSFM")
    translation_averager = TranslationAveraging1DSFM(True, True)
    translations = translation_averager.run_translation_averaging(
        len(rotations), translation_i1_to_i2_dict, rotations)

    logger.info("Done")

    # plot the result
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    for i in range(len(rotations)):
        plot_camera(rotations[i], translations[i], ax, 1.0)
    ax.axis('equal')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_folder', help='path to input images folder')
    parser.add_argument('model_path', help='path to trained model')
    args = parser.parse_args()

    reconstruction_test(args)
