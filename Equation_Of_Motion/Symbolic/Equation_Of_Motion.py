import numpy as np

"""
    Defining static variables and matrices.
"""

# Lengths on the pendulum
l1 = 15
l2 = 1
l3 = 2
l4 = 2
l5 = 1
l6 = 2
l7 = 3

# Radius of each body (since we assume the arms are cylinders and the end is a sphere)
r1 = 1
r2 = 1
r3 = 2

# Mass of each body
m1 = 1
m2 = 1
m3 = 3

# Gravitational acceleration
g = 9.81

# Mass moments of inertia of each body, i about each axis, j. Jij
J11 = 0
J12 = 0
J13 = 0
J21 = 0
J22 = 0
J23 = 0
J31 = 0
J32 = 0
J33 = 0

# Tidsvariabler
time = 0
time_end = 100
dt = 0.1
steps = int((time_end - time) / dt)

# M-matrix
M = np.array([
    [J11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, J12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, J13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0,  m1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0,  m1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0,  m1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, J21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, J22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, J23, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0,  m2, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  m2, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  m2, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, J31, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, J32, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, J33, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  m3, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  m3, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  m3]])

# F-matrix
F = np.array([
    [    0],
    [    0],
    [-m1*g],
    [    0],
    [    0],
    [    0],
    [    0],
    [    0],
    [-m2*g],
    [    0],
    [    0],
    [    0],
    [    0],
    [    0],
    [-m3*g],
    [    0],
    [    0],
    [    0]])


"""
    Defining time-dependant variables and matrices
"""

# Time
time: float

# Angles
theta: float
phi: float

# Angular velocity
theta_dot: float
phi_dot: float

# Angular acceleration
theta_dot_dot: float
phi_dot_dot: float

# D-matrix
def D (theta_dot, phi_dot) -> np.array:
    td = theta_dot
    pd = phi_dot
    return np.array([
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0, -td, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0, td,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0, -(td+pd), 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0, (td+pd),        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0, -(td+pd), 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0, (td+pd),        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0], 
        [0,  0,   0, 0, 0, 0, 0,       0,        0, 0, 0, 0, 0,       0,        0, 0, 0, 0]])


# B-matrix
def B (theta, phi) -> np.array:
    return np.array([
        [                                           0,                          0],
        [                           -l3*np.sin(theta),                          0],
        [                            l3*np.cos(theta),                          0],
        [                                           1,                          0],
        [                                           0,                          0],
        [                                           0,                          0],
        [                                           0,                          0],
        [                      (l3+l4)*np.sin(-theta),       l6*np.sin(theta+phi)],
        [                      (l3+l4)*np.cos(-theta),       l6*np.cos(theta+phi)],
        [                                           1,                          1],
        [                                           0,                          0],
        [                                           0,                          0],
        [                                           0,                          0],
        [ (l3+l4)*np.sin(-theta)-l7*np.sin(theta+phi),  (l6+l7)*np.sin(theta+phi)],
        [ (l3+l4)*np.cos(-theta)+l7*np.cos(theta+phi),  (l6+l7)*np.cos(theta+phi)],
        [                                           1,                          1],
        [                                           0,                          0],
        [                                           0,                          0]])


def B_T (theta, phi) -> np.array:
    return np.transpose(B(theta, phi))


#B-dot-matrix
def B_dot (theta, theta_dot, phi, phi_dot) -> np.array:
    return np.array([
        [                                                                             0,                                               0],
        [                                                   -l3*theta_dot*np.cos(theta),                                               0],
        [                                                   -l3*theta_dot*np.sin(theta),                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                           (l3+l4)*(-theta_dot)*np.cos(-theta),        l6*(theta_dot+phi_dot)*np.cos(theta+phi)],
        [                                          -(l3+l4)*(-theta_dot)*np.sin(-theta),       -l6*(theta_dot+phi_dot)*np.sin(theta+phi)],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [  (l3+l4)*(-theta_dot)*np.cos(-theta)-l7*(theta_dot+phi_dot)*np.cos(theta+phi),   (l6+l7)*(theta_dot+phi_dot)*np.cos(theta+phi)],
        [ -(l3+l4)*(-theta_dot)*np.sin(-theta)-l7*(theta_dot+phi_dot)*np.sin(theta+phi),  -(l6+l7)*(theta_dot+phi_dot)*np.sin(theta+phi)],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0],
        [                                                                             0,                                               0]])


def M_star (theta, phi) -> np.array:
    """
        M* = B_T * M * B
    """
    return np.dot( np.dot( B_T(theta, phi), M ), B(theta, phi))


def M_star_inv (theta, phi) -> np.array:
    return np.linalg.inv(M_star(theta, phi))


def N_star (theta, theta_dot, phi, phi_dot) -> np.array:
    """
        N* = B_T * M * B_dot + B_T * D * M * B
    """
    return np.add( np.dot( np.dot( B_T(theta, phi), M ), B_dot(theta, theta_dot, phi, phi_dot) ), np.dot( np.dot( np.dot( B_T(theta, phi), D(theta_dot, phi_dot) ), M ), B(theta, phi) ) )


def F_star (theta, phi) -> np.array:
    """
        F* = B_T * F
    """
    return np.dot( B_T(theta, phi), F )


def Q (theta_dot, phi_dot):
    return np.array([
        [theta_dot],
        [phi_dot]])


def solve_Q_dot (time, theta, theta_dot, phi, phi_dot) -> np.array:
    """
        Equation of motion solving for Q_dot.
        Q_dot = inv(M*) * (F* - N* * Q)
    """
    return np.subtract( np.dot( M_star_inv(theta, phi), F_star(theta, phi) ), np.dot( N_star(theta, theta_dot, phi, phi_dot), Q(theta_dot, phi_dot) ) )


def integrate_Q_dot (data) -> np.array:
    return solve_Q_dot(theta=data[0], theta_dot=data[1], phi=data[2], phi_dot=[3])

