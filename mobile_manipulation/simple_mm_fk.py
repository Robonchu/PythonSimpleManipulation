import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Link Length
L1 = [0., 0., 0.1]
L2 = [0., 0., 0.1]
L3 = [0., 0., 0.1]
L4 = [0., 0., 0.025]
LINK_LENGTHS = np.array([L1, L2, L3, L4])

# Joint Vector
J1 = [0, 1, 0]
J2 = [0, 1, 0]
J3 = [0, 1, 0]
J4 = [0, 1, 0]
JOINT_VECTORS = np.array([J1, J2, J3, J4])


def skew_mat(vector):
    mat = np.zeros((3, 3))
    mat[0, 1] = -vector[2]
    mat[0, 2] = vector[1]
    mat[1, 0] = vector[2]
    mat[1, 2] = -vector[0]
    mat[2, 0] = -vector[1]
    mat[2, 1] = vector[0]
    return mat


def rodrigues_mat(vector, angle):
    mat = np.eye(3) + skew_mat(vector) * np.sin(angle) + skew_mat(
        vector) @ skew_mat(vector) * (1.0 - np.cos(angle))
    return mat


def calc_mm_fk(base_pos, angles, vectors, lengths):
    assert len(angles) == len(vectors)
    assert len(angles) == len(lengths)
    pos = base_pos
    # R = rodrigues_mat(vectors[0], angles[0])
    R = np.eye(3)
    pos_list = [pos]
    R_list = [R]
    for i in range(len(lengths)):
        R = R @ rodrigues_mat(vectors[i], angles[i])
        R_list.append(R)
        pos = pos + R @ lengths[i]
        pos_list.append(pos)
    return pos, R, pos_list, R_list


""" def calc_fk(angles, vectors, lengths, dof=6):
    iter_num = dof + 1
    pos = [0, 0, 0]
    R = np.eye(3)
    pos_list = np.zeros((dof + 2, 3))
    R_list = np.zeros((dof + 2, 3, 3))
    pos_list[0] = pos
    R_list[0] = R
    # Calculate Forward Kinematics
    for i in range(iter_num):
        pos = pos + R @ lengths[i].T
        R = R @ rodrigues_mat(vectors[i], angles[i])
        pos_list[i + 1] = pos
        R_list[i + 1] = R
    return pos, R, pos_list, R_list
 """


def draw_link_position(pos_list, dof=6):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("myCobotSim", size=20)
    ax.set_xlabel("x", size=10, color="black")
    ax.set_ylabel("y", size=10, color="black")
    ax.set_zlabel("z", size=10, color="black")
    ax.set_xlim3d(0., 0.4)
    ax.set_ylim3d(0., 0.4)
    ax.set_zlim3d(0., 0.4)

    axs = ax.plot([pos_list[0, 0]-0.05, pos_list[0, 0]+0.05],
                  [pos_list[0, 1], pos_list[0, 1]],
                  [pos_list[0, 2], pos_list[0, 2]],
                  color='blue',
                  linewidth=10)

    for i in range(len(pos_list)-1):
        axs += ax.plot([pos_list[i, 0], pos_list[i + 1, 0]],
                       [pos_list[i, 1], pos_list[i + 1, 1]],
                       [pos_list[i, 2], pos_list[i + 1, 2]],
                       color='blue')
    ax.scatter(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], color='red')
    plt.show()


def run():
    base_pos = [np.random.rand()*0.1, 0., 0.]
    rad1 = np.random.rand() * np.pi / 2.0
    rad2 = np.random.rand() * np.pi / 2.0
    rad3 = np.random.rand() * np.pi / 2.0
    rad4 = np.random.rand() * np.pi / 2.0
    angles = np.array([rad1, rad2, rad3, rad4])
    pos, R, pos_list, R_list = calc_mm_fk(
        base_pos, angles, JOINT_VECTORS, LINK_LENGTHS)
    # import pdb
    # pdb.set_trace()
    draw_link_position(np.array(pos_list))


if __name__ == "__main__":
    run()
