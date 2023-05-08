from collections import defaultdict
from enum import Enum
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
import gtsam
import numpy as np
from loguru import logger
from gtsam import (
    BinaryMeasurementsPoint3,
    BinaryMeasurementPoint3,
    BinaryMeasurementsUnit3,
    BinaryMeasurementUnit3,
    MFAS,
    Point3,
    Pose3,
    Rot3,
    symbol_shorthand,
    TranslationRecovery,
    Unit3,
)
import matplotlib.pyplot as plt


# https://github.com/borglab/gtsfm/blob/master/gtsfm/averaging/translation/averaging_1dsfm.py

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01
HUBER_LOSS_K = 1.345  # default value from GTSAM
C = symbol_shorthand.A  # for camera translation variables


def get_valid_measurements_in_world_frame(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wRi_list: List[Optional[Rot3]]
) -> Tuple[Dict[Tuple[int, int], Unit3], Set[int]]:
    """Returns measurements for which both cameras have valid rotations, transformed to the world frame """
    w_i2Ui1_dict = {}
    valid_cameras: Set[int] = set()
    for (i1, i2), i2Ui1 in i2Ui1_dict.items():
        wRi2 = wRi_list[i2]
        if i2Ui1 is not None and wRi2 is not None:
            w_i2Ui1_dict[(i1, i2)] = Unit3(wRi2.rotate(i2Ui1.point3()))
            valid_cameras.add(i1)
            valid_cameras.add(i2)
    return w_i2Ui1_dict, valid_cameras


class TranslationAveraging1DSFM:
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(
        self,
        robust_measurement_noise: bool = True,
        reject_outliers: bool = True,
    ) -> None:
        """Initializes the 1DSFM averaging instance."""
        self._robust_measurement_noise = robust_measurement_noise
        self._outlier_weight_threshold = 0.125
        self._reject_outliers = reject_outliers
        # we will not use tracks, since we don't have matches

    def run_translation_averaging(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        scale_factor: float = 1.0,
    ) -> List[Optional[Pose3]]:
        logger.info("Running translation averaging on %d unit translations" % len(i2Ui1_dict))

        # get relative translations, in world coordinate : (translation_2 - translation_1)
        w_i2Ui1_dict, valid_cameras = get_valid_measurements_in_world_frame(i2Ui1_dict, wRi_list)

        # TODO: add outlier rejection

        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
        if self._robust_measurement_noise:
            huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
            noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)
        w_i1Ui2_measurements = BinaryMeasurementsUnit3()
        for (i1, i2), w_i2Ui1 in w_i2Ui1_dict.items():
            w_i1Ui2_measurements.append(BinaryMeasurementUnit3(C(i1), C(i2), w_i2Ui1, noise_model))

        # process averaging
        algorithm = TranslationRecovery()
        wti_values = algorithm.run(w_i1Ui2_measurements, scale_factor)

        # Transforms the result to a list of Point3 objects.
        wti_list: List[Optional[Point3]] = [None] * num_images
        for i in range(num_images):
            if wRi_list[i] is not None and wti_values.exists(C(i)):
                wti_list[i] = wti_values.atPoint3(C(i))

        return wti_list

# python reconstruction/translation_averaging.py
def translation_averaging_test():
    # make a simulation dataset, and test the rotation averaging result
    test_size = 50
    rotation_gt = []
    translation_gt = []
    np.random.seed(2023)
    logger.info("Test with %d images." % test_size)

    # generate random gt rotation
    for i in range(test_size):
        angle = np.pi * np.random.rand(3)
        rotation_gt.append(Rot3.RzRyRx(angle[0], angle[1], angle[2]))
        position = 100.0 * np.random.rand(3)
        translation_gt.append(Point3(position[0], position[1], position[2]))

    # generate random matches
    translation_i1_to_i2_dict = {}
    # * add sequence match
    for i in range(test_size - 1):
        translation_1_to_2 = translation_gt[i + 1] - translation_gt[i]
        translation_i1_to_i2_dict[
            (i, i + 1)
        ] = Unit3(rotation_gt[i + 1].inverse().rotate(translation_1_to_2))
    # * add random match
    for i in range(test_size * 5):
        id1_id2 = np.random.randint(test_size, size=2)
        if id1_id2[0] == id1_id2[1] :
            continue
        translation_1_to_2 = translation_gt[id1_id2[1]] - translation_gt[id1_id2[0]]
        translation_i1_to_i2_dict[
            (id1_id2[0], id1_id2[1])
        ] = Unit3(rotation_gt[id1_id2[1]].inverse().rotate(translation_1_to_2))

    translation_averager = TranslationAveraging1DSFM(True, True)
    translations_estimation = translation_averager.run_translation_averaging(
        len(rotation_gt), translation_i1_to_i2_dict, rotation_gt)

    # check the result
    if len(translations_estimation) != len(translation_gt):
        logger.error("Size wrong.")

    rescale = np.linalg.norm(translation_gt[1] - translation_gt[0]) / \
              np.linalg.norm(translations_estimation[1] - translations_estimation[0])
    offset = translation_gt[0] - rescale * translations_estimation[0]
    logger.info("  rescale factor : %f" % rescale)
    logger.info(f"  offset : {offset}")
    for i in range(len(translation_gt)):
        rescaled_estimation = rescale * translations_estimation[i] + offset
        error_norm = np.linalg.norm(translation_gt[i] - rescaled_estimation)
        if error_norm > 1e-5:
            logger.error(f"{i} has large error {error_norm}")


    logger.info("Test Pass")
    # plot the result
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    for i in range(len(translation_gt)):
        plot_camera(rotation_gt[i], translations_estimation[i], ax)

    plt.show()


def plot_camera(rotation, translation, ax, cam_size = 0.1):
    def transform_pt(point):
        return rotation.rotate(point) + translation
    pt_o = transform_pt(Point3(0, 0, 0))
    pt_1 = transform_pt(Point3(cam_size, cam_size, cam_size))
    pt_2 = transform_pt(Point3(-cam_size, cam_size, cam_size))
    pt_3 = transform_pt(Point3(-cam_size, -cam_size, cam_size))
    pt_4 = transform_pt(Point3(cam_size, -cam_size, cam_size))

    color = np.random.rand(3)
    def get_axis(axis):
        return [pt_1[axis], pt_2[axis], pt_3[axis], pt_4[axis], pt_1[axis]]
    ax.plot3D(get_axis(0), get_axis(1), get_axis(2), c=color)
    def plot_line(pt_t):
        ax.plot3D([pt_o[0], pt_t[0]], [pt_o[1], pt_t[1]], [pt_o[2], pt_t[2]], c=color)
    plot_line(pt_1)
    plot_line(pt_2)
    plot_line(pt_3)
    plot_line(pt_4)


if __name__ == "__main__":
    translation_averaging_test()
