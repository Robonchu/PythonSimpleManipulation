import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Link Angle Vector
a1 = [0, 0, 1]
a2 = [0, 1, 0]
a3 = [0, 1, 0]
a4 = [0, 1, 0]
a5 = [0, 0, 1]
a6 = [1, 0, 0]
a7 = [0, 0, 0]
## Gripper
a8 = [0, 0, 0]
a9 = [0, 0, 0]
a10 = [0, 0, 0]
a11 = [0, 0, 0]

# vector_list = [a1, a2, a3, a4, a5, a6, a7]
vector_list = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]

# Link Length
b1 = [0., 0., 0.06114]
b2 = [0., 0., 0.07042]
b3 = [0., 0., 0.1104]
b4 = [0., 0., 0.0960]
b5 = [0., 0.06639, 0.]
b6 = [0., 0., 0.07318]
b7 = [0.0436, 0., 0.]
## Gripper
b8 = [0, -0.02, 0.]
b9 = [0., 0.04, 0.]
b10 = [0.02, 0., 0.]
b11 = [0.0, -0.04, 0.]

# length_list = np.array([b1, b2, b3, b4, b5, b6, b7])
length_list = np.array([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11])


def skew_mat(vector):
    mat = np.zeros((3, 3))
    mat[0, 1] = -vector[2]
    mat[0, 2] = vector[1]
    mat[1, 0] = vector[2]
    mat[1, 2] = -vector[0]
    mat[2, 0] = -vector[1]
    mat[2, 1] = vector[0]
    return mat


def rodrigues(vector, angle):
    mat = np.eye(3) + skew_mat(vector) * np.sin(angle) + skew_mat(
        vector) @ skew_mat(vector) * (1.0 - np.cos(angle))
    return mat


def CalcFK(angle_list, vector_list, length_list, dof=6):
    calc_num = dof + 1
    pos = [0, 0, 0]
    R = np.eye(3)
    pos_list = [pos]
    R_list = [R]
    # Calculate Forward Kinematics
    for i in range(calc_num):
        pos = pos + R @ length_list[i].T
        R = R @ rodrigues(vector_list[i], angle_list[i])
        pos_list.append(pos)
        R_list.append(R)
    return pos, R, pos_list, R_list


def CalcJacobi(vector_list, pos_list, rot_list, dof=6):
    # memo: len(pos_list)= 6 or 7, vector_list = 6, rot_list = 6
    J = np.zeros((dof, dof))
    for i in range(dof):
        delta_angle = rot_list[i] @ vector_list[i]
        delta_pos = skew_mat(delta_angle) @ (pos_list[-1] - pos_list[i])
        J[3:, i] = delta_angle
        J[:3, i] = delta_pos
    return J


def CalcErr(target_pos, target_rot, current_pos, current_rot):
    R = target_rot - current_rot
    pos_err = np.square(target_pos - current_pos)
    theta = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.0)
    ln_rot = theta / 2 * np.sin(theta) * np.array(
        [[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])
    rot_err = np.square(ln_rot)
    return pos_err, rot_err


def CalcIK(angle_list,
           vector_list,
           length_list,
           target_pos,
           target_rot,
           threshold,
           max_itr=1000):
    alpha = 1.0

    for i in range(max_itr):
        current_pos, current_rot, pos_list, rot_list = CalcFK(
            angle_list, vector_list, length_list)
        J = CalcJacobi(vector_list, pos_list[1:], rot_list[1:])
        pos_err, rot_err = CalcErr(target_pos, target_rot, current_pos,
                                   current_rot)
        err = np.concatenate([pos_err, rot_err[:, 0]])
        if np.linalg.norm(err) < threshold:
            break
        delta_angle = alpha * (np.linalg.inv(J) @ err)
        angle_list = np.array(angle_list) + np.concatenate([delta_angle, [0]])
    return angle_list


if __name__ == "__main__":
    ARM_NUM = 7
    GRIPPER_NUM = 4
    TOTAL_NUM = ARM_NUM + GRIPPER_NUM
    pos = [0, 0, 0]
    R = np.eye(3)
    # rad1 = np.random.rand() * np.pi / 2.0
    # rad2 = np.random.rand() * np.pi / 2.0
    # rad3 = np.random.rand() * np.pi / 2.0
    # rad4 = np.random.rand() * np.pi / 2.0
    # rad5 = np.random.rand() * np.pi / 2.0
    # rad6 = np.random.rand() * np.pi / 2.0

    rad1 = 0.
    rad2 = np.pi / 5.
    rad3 = np.pi / 5
    rad4 = -np.pi / 5 * 2
    rad5 = 0.
    rad6 = 0.
    # angle_list = [rad1, rad2, rad3, rad4, rad5, rad6, 0]
    angle_list = [rad1, rad2, rad3, rad4, rad5, rad6, 0, 0, 0, 0, 0]
    pos, R, pos_list, R_list = CalcFK(angle_list, vector_list, length_list)
    import pdb
    pdb.set_trace()

    target_rot = np.eye(3)
    target_pos = np.array([0.21766459, 0.06639, 0.28280459])
    threshold = 0.00001
    angle_list = CalcIK(angle_list[:7], vector_list, length_list, target_pos,
                        target_rot, threshold)

    import pdb
    pdb.set_trace()
    '''
    pos_list = [pos]
    R_list = [R]

    pos_x = [pos[0]]
    pos_y = [pos[1]]
    pos_z = [pos[2]]
    # Calculate Forward Kinematics
    for i in range(TOTAL_NUM):
        pos = pos + R @ length_list[i].T
        R = R @ rodrigues(vector_list[i], angle_list[i])
        pos_list.append(pos)
        R_list.append(R)
        pos_x.append(pos[0])
        pos_y.append(pos[1])
        pos_z.append(pos[2])

    # Figureを追加
    fig = plt.figure(figsize=(8, 8))
    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')
    # Axesのタイトルを設定
    ax.set_title("myCobotSim", size=20)

    # 軸ラベルを設定
    ax.set_xlabel("x", size=10, color="black")
    ax.set_ylabel("y", size=10, color="black")
    ax.set_zlabel("z", size=10, color="black")

    ax.set_xlim3d(-0.3, 0.3)
    ax.set_ylim3d(-0.3, 0.3)
    ax.set_zlim3d(0, 0.6)

    ax.scatter(pos_x[:ARM_NUM], pos_y[:ARM_NUM], pos_z[:ARM_NUM], color='red')
    ax.scatter(pos_x[ARM_NUM:],
               pos_y[ARM_NUM:],
               pos_z[ARM_NUM:],
               color='green')
    for i in range(ARM_NUM):
        if i == 0:
            ax.plot([pos_x[i], pos_x[i + 1]], [pos_y[i], pos_y[i + 1]],
                    [pos_z[i], pos_z[i + 1]],
                    color='blue',
                    linewidth=10)
        else:
            ax.plot([pos_x[i], pos_x[i + 1]], [pos_y[i], pos_y[i + 1]],
                    [pos_z[i], pos_z[i + 1]],
                    color='blue')
    for i in range(ARM_NUM, TOTAL_NUM - 1):
        ax.plot([pos_x[i], pos_x[i + 1]], [pos_y[i], pos_y[i + 1]],
                [pos_z[i], pos_z[i + 1]],
                color='green')
    ax.plot([pos_x[ARM_NUM + 1], pos_x[TOTAL_NUM]],
            [pos_y[ARM_NUM + 1], pos_y[TOTAL_NUM]],
            [pos_z[ARM_NUM + 1], pos_z[TOTAL_NUM]],
            color='green')

    # ax.plot([pos_x[ARM_NUM], pos_x[ARM_NUM+1]], [pos_y[ARM_NUM], pos_y[ARM_NUM+1]], [pos_z[ARM_NUM], pos_z[ARM_NUM+1]], color='green')
    # for i in range(GRIPPER_NUM):
    #     ax.plot([pos_x[i], pos_x[i + 1]], [pos_y[i], pos_y[i + 1]],
    #             [pos_z[i], pos_z[i + 1]],
    #             color='blue')
    plt.show()
    '''
