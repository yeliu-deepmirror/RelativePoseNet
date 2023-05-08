from typing import Dict, List, Optional, Set, Tuple
import gtsam
import numpy as np
from loguru import logger
from gtsam import (
    BetweenFactorPose3,
    BetweenFactorPose3s,
    LevenbergMarquardtParams,
    Pose3,
    Rot3,
    ShonanAveraging3,
    ShonanAveragingParameters3,
)

TWOVIEW_ROTATION_SIGMA = 1
POSE3_DOF = 6

# https://github.com/borglab/gtsfm
# https://github.com/borglab/gtsfm/blob/master/gtsfm/averaging/rotation/shonan.py
class ShonanRotationAveraging:
    """Performs Shonan rotation averaging."""
    def __init__(self) -> None:
        """
        Note: `p_min` and `p_max` describe the minimum and maximum relaxation rank.
        """
        self._p_min = 5
        self._p_max = 30

    def __get_shonan_params(self) -> ShonanAveragingParameters3:
        lm_params = LevenbergMarquardtParams.CeresDefaults()
        shonan_params = ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(False)
        shonan_params.setCertifyOptimality(True)
        return shonan_params

    def __between_factors_from_2view_relative_rotations(
        self, rotation_i1_to_i2_dict: Dict[Tuple[int, int], Rot3], old_to_new_idxs: Dict[int, int]
    ) -> BetweenFactorPose3s:
        """Create between factors from relative rotations computed by the 2-view estimator."""
        # TODO: how to weight the noise model on relative rotations compared to priors?
        noise_model = gtsam.noiseModel.Isotropic.Sigma(POSE3_DOF, TWOVIEW_ROTATION_SIGMA)

        between_factors = BetweenFactorPose3s()

        for (i1, i2), i2Ri1 in rotation_i1_to_i2_dict.items():
            if i2Ri1 is not None:
                # ignore translation during rotation averaging
                i2Ti1 = Pose3(i2Ri1, np.zeros(3))
                i2_ = old_to_new_idxs[i2]
                i1_ = old_to_new_idxs[i1]
                between_factors.append(BetweenFactorPose3(i2_, i1_, i2Ti1, noise_model))

        return between_factors

    def _run_with_consecutive_ordering(
        self, num_connected_nodes: int, between_factors: BetweenFactorPose3s
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph w/ N keys ordered consecutively [0,...,N-1]. """

        logger.info(
            f"Running Shonan with {len(between_factors)} constraints on {num_connected_nodes} nodes")
        shonan = ShonanAveraging3(between_factors, self.__get_shonan_params())

        initial = shonan.initializeRandomly()
        logger.info("Initial cost: %.5f" % shonan.cost(initial))
        result, _ = shonan.run(initial, self._p_min, self._p_max)
        logger.info("Final cost: %.5f" % shonan.cost(result))

        wRi_list_consecutive = [None] * num_connected_nodes
        for i in range(num_connected_nodes):
            if result.exists(i):
                wRi_list_consecutive[i] = result.atRot3(i)

        return wRi_list_consecutive

    def _nodes_with_edges(
        self, rotation_i1_to_i2_dict: Dict[Tuple[int, int], Optional[Rot3]]
    ) -> Set[int]:
        """Gets the nodes with edges which are to be modelled as between factors."""
        unique_nodes_with_edges = set()
        for (i1, i2) in rotation_i1_to_i2_dict.keys():
            unique_nodes_with_edges.add(i1)
            unique_nodes_with_edges.add(i2)
        return unique_nodes_with_edges

    def run_rotation_averaging(
        self,
        num_images: int,
        rotation_i1_to_i2_dict: Dict[Tuple[int, int], Optional[Rot3]],
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.

        Note: functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
        All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
        camera in a single, global coordinate frame.

        Args:
            num_images: number of images. Since we have one pose per image, it is also the number of poses.
            rotation_i1_to_i2_dict: relative rotations for each image pair-edge as dictionary (i1, i2):
                rotation from frame1 to frame2.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system), or where the camera pose had no valid observation
                in the input to run_rotation_averaging().
        """
        if len(rotation_i1_to_i2_dict) == 0:
            logger.warning("Shonan cannot proceed: No cycle-consistent triplets found after filtering.")
            wRi_list = [None] * num_images
            return wRi_list

        nodes_with_edges = sorted(list(self._nodes_with_edges(rotation_i1_to_i2_dict)))
        old_to_new_idxes = {old_idx: i for i, old_idx in enumerate(nodes_with_edges)}

        between_factors: BetweenFactorPose3s = self.__between_factors_from_2view_relative_rotations(
            rotation_i1_to_i2_dict, old_to_new_idxes
        )
        wRi_list_subset = self._run_with_consecutive_ordering(
            num_connected_nodes=len(nodes_with_edges), between_factors=between_factors
        )

        wRi_list = [None] * num_images
        for remapped_i, original_i in enumerate(nodes_with_edges):
            wRi_list[original_i] = wRi_list_subset[remapped_i]

        return wRi_list


# python reconstruction/rotation_averaging.py
def rotation_averaging_test():
    # make a simulation dataset, and test the rotation averaging result
    test_size = 50
    rotation_gt = []
    np.random.seed(2023)
    logger.info("Test with %d images." % test_size)

    # generate random gt rotation
    for i in range(test_size):
        angle = np.pi * np.random.rand(3)
        rotation_gt.append(Rot3.RzRyRx(angle[0], angle[1], angle[2]))

    # generate random matches
    rotation_i1_to_i2_dict = {}
    # * add sequence match
    for i in range(test_size - 1):
        rotation_1 = rotation_gt[i]
        rotation_2 = rotation_gt[i + 1]
        rotation_i1_to_i2_dict[(i, i + 1)] = rotation_2.inverse().compose(rotation_1)
    # * add random match
    for i in range(test_size):
        id1_id2 = np.random.randint(test_size, size=2)
        if id1_id2[0] == id1_id2[1] :
            continue
        rotation_i1_to_i2_dict[
            (id1_id2[0], id1_id2[1])
        ] = rotation_gt[id1_id2[1]].inverse().compose(rotation_gt[id1_id2[0]])

    logger.info("Make %d matches." % len(rotation_i1_to_i2_dict))

    rotation_averaging = ShonanRotationAveraging()
    rotation_estimation = rotation_averaging.run_rotation_averaging(test_size, rotation_i1_to_i2_dict)

    # check the result
    if len(rotation_estimation) != len(rotation_gt):
        logger.error("Size wrong.")

    offset = rotation_estimation[0].compose(rotation_gt[0].inverse())
    for i in range(test_size):
        error = rotation_estimation[i].inverse().compose(offset).compose(rotation_gt[i])
        error_norm = np.linalg.norm(Rot3.Logmap(error))
        if error_norm > 1e-5:
            logger.error(f"{i} has large error {error_norm}")

    logger.info("Test Pass")


if __name__ == "__main__":
    rotation_averaging_test()
