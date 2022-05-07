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
J0 = [0, 0, 1]
J1 = [0, 1, 0]
J2 = [0, 1, 0]
J3 = [0, 1, 0]
J4 = [0, 1, 0]
# JOINT_VECTORS = np.array([J1, J2, J3, J4])
JOINT_VECTORS = np.array([J0, J1, J2, J3, J4])


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


def calc_jacobi(vector_list, pos_list, rot_list):
    # Note: pos_list[0], rot_list[0] is mobile_base info.
    assert len(vector_list) == len(pos_list)
    assert len(vector_list) == len(rot_list)
    J = np.zeros((6, len(vector_list)))
    for i in range(len(vector_list)):
        delta_angle = rot_list[i] @ vector_list[i]
        J[3:, i] = delta_angle
        delta_pos = skew_mat(delta_angle) @ (pos_list[-1] - pos_list[i])
        J[:3, i] = delta_pos
    return J


def calc_err(target_pos, target_rot, current_pos, current_rot, rot_th=0.0001):
    pos_err = np.square(target_pos - current_pos)
    R_err = target_rot - current_rot
    theta = np.arccos((R_err[0, 0] + R_err[1, 1] + R_err[2, 2] - 1) / 2.0)
    ln_rot = theta / (2 * np.sin(theta)) * np.array(
        [[R_err[2, 1] - R_err[1, 2]],
            [R_err[0, 2] - R_err[2, 0]],
            [R_err[1, 0] - R_err[0, 1]]])
    rot_err = np.square(ln_rot)
    return pos_err, rot_err


def calc_mm_ik(base_pos, angle_list,
               vector_list,
               length_list,
               target_pos,
               target_rot,
               threshold,
               max_itr=50000):
    alpha = 1.0
    # alpha = np.array([0.001, 1.0, 1.0, 1.0, 1.0])
    for i in range(max_itr):
        current_pos, current_rot, pos_list, rot_list = calc_mm_fk(
            base_pos, angle_list, vector_list[1:], length_list)
        J = calc_jacobi(vector_list, pos_list, rot_list)
        pos_err, rot_err = calc_err(target_pos, target_rot, current_pos,
                                    current_rot)
        err = np.concatenate([pos_err, rot_err[:, 0]])
        print("iter:", i)
        print("err:", np.linalg.norm(err))
        if np.linalg.norm(err) < threshold:
            break
        elif np.linalg.norm(err) > 100.:
            return None, None

        # lam = np.array([1., 0.1, 0.1, 0.1, 0.1])
        lam = 0.1
        Jninv = np.linalg.inv((J.T @ J + lam * np.eye(len(J[0])))) @ J.T
        # delta_angle = alpha * (np.linalg.pinv(J) @ err)
        delta_base_angle = alpha * (Jninv @ err)
        delta_base = np.array([delta_base_angle[1], 0, 0])
        delta_angle = delta_base_angle[1:]
        base_pos += delta_base
        angle_list = np.array(angle_list)
        angle_list = np.array(angle_list) + delta_angle
    return base_pos, angle_list


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
    init_posx = 0.05
    init_base_pos = [init_posx, 0., 0.]
    rad1 = np.pi/8
    rad2 = np.pi/8
    rad3 = np.pi/8
    rad4 = np.pi/8
    init_angles = np.array([rad1, rad2, rad3, rad4])
    init_pos, init_R, init_pos_list, init_R_list = calc_mm_fk(
        init_base_pos, init_angles, JOINT_VECTORS[1:], LINK_LENGTHS)

    target_posx = 0.06
    target_base_pos = [target_posx, 0., 0.]
    rad1 = np.pi/8
    rad2 = np.pi/8
    rad3 = np.pi/8
    rad4 = np.pi/7
    target_angles = np.array([rad1, rad2, rad3, rad4])
    target_pos, target_R, target_pos_list, target_R_list = calc_mm_fk(
        target_base_pos, target_angles, JOINT_VECTORS[1:], LINK_LENGTHS)

    base_pos, angle_list = calc_mm_ik(init_base_pos, init_angles,
                                      JOINT_VECTORS, LINK_LENGTHS, target_pos,
                                      target_R, 0.000005)

    pos, R, pos_list, R_list = calc_mm_fk(
        base_pos, angle_list, JOINT_VECTORS[1:], LINK_LENGTHS)

    print("base_pos:", base_pos, "angle_list:", angle_list)
    print("init_pos:", init_pos, "init_R:", init_R)
    print("target_pos:", target_pos, "target_R:", target_R)
    print("final_pos:", pos, "final_R:", R)
    draw_link_position(np.array(pos_list))


if __name__ == "__main__":
    run()
