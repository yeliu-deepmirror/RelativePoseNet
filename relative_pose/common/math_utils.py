import numpy as np

def intrinsics_matrix_to_param(intri_matrix):
    return np.array([intri_matrix[0, 0], intri_matrix[1, 1], intri_matrix[0, 2], intri_matrix[1, 2]])

def cross_product_matrix(point):
    return np.array([[0, -point[2], point[1]],
                     [point[2], 0, -point[0]],
                     [-point[1], point[0], 0]])


def relative_pose_to_essential_matrix(rotation_matrix, translation):
    # Calculate Essential Matrix E that
    # p2 = R * p1 + T => p2^T * E * p1 = 0
    return cross_product_matrix(translation).dot(rotation_matrix)


# return four cases
# TODO(yeliu): check the output
def fundamental_matrix_to_pose(intri_matrix_1, intri_matrix_2, fundamental_1_to_2):
    essential_1_to_2 = ((np.transpose(intri_matrix_2)).dot(fundamental_1_to_2)).dot(intri_matrix_1)

    # essential matrix to relative pose
    u, s, vh = np.linalg.svd(essential_1_to_2, full_matrices=True)
    w = np.zeros(3, 3)
    w[0, 1] = -1
    w[1, 0] = 1
    w[2, 2] = 1

    translation = u[:, 2]
    rotation_1 = (u.dot(w)).dot(vh)
    if np.linalg.det(rotation_1) < 0:
        rotation_1 = -rotation_1
    rotation_2 = (u.dot(np.transpose(w))).dot(vh)
    if np.linalg.det(rotation_2) < 0:
        rotation_2 = -rotation_2

    # TODO check the poses
    return translation, -translation, rotation_1, rotation_2
