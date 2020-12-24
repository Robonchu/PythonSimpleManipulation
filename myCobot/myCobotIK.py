import numpy as np
import sympy as sp

# Link Angle Vector
a1 = [0, 0, 1]
a2 = [0, 1, 0]
a3 = [0, 1, 0]
a4 = [0, 1, 0]
a5 = [0, 0, 1]
a6 = [1, 0, 0]
a7 = [0, 0, 0]
vector_list = [a1, a2, a3, a4, a5, a6, a7]

# Link Length
# b1 = [0., 0., 0.06114]
# b2 = [0., 0., 0.07042]
# b3 = [0., 0., 0.1104]
# b4 = [0., 0., 0.0960]
# b5 = [0., 0.06639, 0.]
# b6 = [0., 0., 0.07318]
# b7 = [0.0436, 0., 0.]
# length_list = sp.Matrix([b1, b2, b3, b4, b5, b6, b7])

# Sympy/Numpy
# ref: https://qiita.com/tibigame/items/539f4f9ed0b6d8d76946


def rot_x(phi):
    r = sp.Matrix([[1., 0., 0], [0., sp.cos(phi), -sp.sin(phi)],
                   [0., sp.sin(phi), sp.cos(phi)]])
    return r


def rot_y(theta):
    r = sp.Matrix([[sp.cos(theta), 0., sp.sin(theta)], [0., 1., 0.],
                   [-sp.sin(theta), 0., sp.cos(theta)]])
    return r


def rot_z(psi):
    r = sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0.],
                   [sp.sin(psi), sp.cos(psi), 0.], [0., 0., 1.]])
    return r


def skew_mat(vector):
    mat = sp.Matrix([[0., -vector[2], vector[1]], [vector[2], 0., -vector[0]],
                     [-vector[1], vector[0], 0.]])
    return mat


def rodrigues(vector, angle):
    mat = sp.eye(3) + skew_mat(vector) * sp.sin(angle) + skew_mat(
        vector) * skew_mat(vector) * (1.0 - sp.cos(angle))
    return mat


if __name__ == "__main__":
    sp.init_printing()
    sp.var('x, y, z, roll, pitch, yaw, ang1, ang2, ang3, ang4, ang5, ang6')
    sp.var('b1, b2, b3, b4, b5, b6, b7')
    c1 = [0., 0., b1]
    c2 = [0., 0., b2]
    c3 = [0., 0., b3]
    c4 = [0., 0., b4]
    c5 = [0., b5, 0.]
    c6 = [0., 0., b6]
    c7 = [b7, 0., 0.]
    length_list = sp.Matrix([c1, c2, c3, c4, c5, c6, c7])

    # sp.var('x, y, a, b, c, g, h')
    # eq5=sp.Eq(y, a*x**2+b*x+c)
    # eq6=sp.Eq(y, g*x+h)
    # print(eq5)
    # sol = sp.solve([eq5, eq6], [x, y])
    # import pdb;pdb.set_trace()
    # print(sol)

    a1 = [0, 0, 1]
    a2 = [0, 1, 0]
    a3 = [0, 1, 0]
    # RyRzRx
    a4 = [0, 1, 0]
    a5 = [0, 0, 1]
    a6 = [1, 0, 0]
    a7 = [0, 0, 0]
    vector_list = [a1, a2, a3, a4, a5, a6, a7]

    pos = sp.Matrix([[0, 0, 0]]).T
    R = sp.eye(3)
    angle_list = [ang1, ang2, ang3, ang4, ang5, ang6, 0]
    for i in range(7):
        pos = pos + R * length_list[i, :].T
        R = R * rodrigues(vector_list[i], angle_list[i])

    pos = sp.simplify(pos)
    R = sp.simplify(R)

    # Rrpy = rot_z(yaw) * rot_y(pitch) * rot_x(roll)
    # Rrpy = rot_y(pitch) * rot_z(yaw) * rot_x(roll)
    # Rrpy = sp.simplify(Rrpy)

    psi = sp.asin(R[1, 0])
    phi = sp.atan2(-R[1, 2], R[1, 1])
    theta = sp.atan2(R[2, 0], -R[0, 0])

    # import pdb
    # pdb.set_trace()
    # phi = sp.atan2(R[2, 1], R[2, 2])
    # theta = sp.asin(-R[2, 0])
    # psi = sp.atan2(R[1, 0], R[0, 0])

    phi = sp.simplify(phi)
    theta = sp.simplify(theta)
    psi = sp.simplify(psi)

    eq1 = sp.Eq(roll, phi)
    eq2 = sp.Eq(pitch, theta)
    eq3 = sp.Eq(yaw, psi)

    # import pdb
    # pdb.set_trace()

    # sol = sp.solve([eq1, eq2, eq3], [ang4, ang5, ang6])

    import pdb
    pdb.set_trace()
    eq10 = sp.Eq(x, pos[0])
    eq11 = sp.Eq(y, pos[1])
    eq12 = sp.Eq(z, pos[2])

    eq1 = sp.Eq(roll, phi)
    eq2 = sp.Eq(pitch, theta)
    eq3 = sp.Eq(yaw, psi)

    sol = sp.solve([eq1, eq2, eq3, eq10, eq11, eq12],
                   [ang1, ang2, ang3, ang4, ang5, ang6])

    import pdb
    pdb.set_trace()
    # eq1 = sp.Eq(R[0, 0], Rrpy[0, 0])
    # eq2 = sp.Eq(R[0, 1], Rrpy[0, 1])
    # eq3 = sp.Eq(R[0, 2], Rrpy[0, 2])
    # eq4 = sp.Eq(R[1, 0], Rrpy[1, 0])
    # eq5 = sp.Eq(R[1, 1], Rrpy[1, 1])
    # eq6 = sp.Eq(R[1, 2], Rrpy[1, 2])
    # eq7 = sp.Eq(R[2, 0], Rrpy[2, 0])
    # eq8 = sp.Eq(R[2, 1], Rrpy[2, 1])
    # eq9 = sp.Eq(R[2, 2], Rrpy[2, 2])
    import pdb
    pdb.set_trace()

    sol = sp.solve(
        [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12],
        [x, y, z, roll, pitch, yaw])
    import pdb
    pdb.set_trace()
