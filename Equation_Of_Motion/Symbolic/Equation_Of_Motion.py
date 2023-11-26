import numpy as np

"""
    Defining static variables and matrices.
"""

# Lengths on the pendulum
# Unit: [m]
l1 = 1
l2 = 0.08
l3 = 0.1
l4 = 0.1
l5 = 0.08
l6 = 0.1
l7 = 0.2

# Radius of each body (since we assume the arms are cylinders and the end is a sphere)
# Unit: [m]
r1 = 0.03
r2 = 0.03
r3 = 0.05

# Radius of joints (for friction)
# Unit: [m]
jr1 = 0.05
jr2 = 0.05

# Mass of each body
# Unit [kg]
m1 = 0.1
m2 = 0.1
m3 = 0.2

# Gravitational acceleration
# Unit: [m/s^2]
g = 9.81

# Drag coefficients
c1 = 0.82
c2 = 0.82
c3 = 0.5

# Air density
# Unit [kg/m^3]
air_density = 1.293

# Friction coefficient
fric = 0.005

# Variable values for whether to include air resistance and friction, and what kind to include.
air_resistance_type: int = 0 # 0 = no air resistance. 1 = simple, "closest-body" air resistance. 2 = complex, "all bodies effect" air resistance.
friction_type: int = 0 # 0 = no friction. 1 = simple friction at a point in the joint dependent on the weight of the bodies.

# Mass moments of inertia of each body, i about each axis, j. Jij
# Formulas taken from wikipedia page about mass moments of inertia og common shapes.
# Unit: [kg/m^2]
J11 = 1/12 * m1 * (3*r1**2 + (l3 + l4)**2)
J12 = 1/2 * m1 * r1**2
J13 = J11
J21 = 1/12 * m2 * (3*r2**2 + (l6 + l7-r3)**2)
J22 = 1/2 * m2 * r2**2
J23 = J21
J31 = 2/3 * m3 * r3**2
J32 = J31
J33 = J31

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


# Air resistance force as moments.
# Unit: [Nm]
def air_resistance_torque (theta, phi, theta_dot, phi_dot) -> list[float]:

    # Return variables
    air_res_theta_dot: float
    air_res_phi_dot: float

    # Determine which direction the force should act depending on what direction the angular velocities have.
    if not theta_dot == 0:
        dir_theta = (-1) * theta_dot / np.abs(theta_dot)
    else:
        dir_theta = 0

    if not phi_dot == 0:
        dir_phi = (-1) * phi_dot / np.abs(phi_dot)
    else:
        dir_phi = 0

    # Determine which type of air resistance to use for the intergration. Default (else) is 0 (no air resistance)
    if air_resistance_type == 1:
        # Use simple air resistance where each variable angle, theta and phi, gets affected by only the body's, in which they belong to, drag force from the respective angular velocity.
        air_res_theta_dot = dir_theta * (2/3 * (1/2 * air_density * c1 * (theta_dot * (l3 + l4))**2) * (l3+l4))
        air_res_phi_dot   = dir_phi * (2/3 * (1/2 * air_density * c2 * (phi_dot   * (l6 + l7 - r3))**2) * (l6+l7-r3)) + dir_phi * (1/2 * air_density * c3 * (phi_dot * (l6+l7))**2) * (l6+l7)

    elif air_resistance_type == 2:
        # Use complex air resistance where each variable angle, theta and phi gets affected by the drag force of each body "further out" on the model 
        # and affected by each angular velocity "further in" on the model as well.
        air_res_theta_dot = dir_theta * ( 2/3 * (l3 + l4 + (l6 + l7)*np.cos(phi)) * (1/2 * c1 * air_density * (l3 + l4 + (l6 + l7 - r3) * np.cos(phi))) )
        air_res_phi_dot = dir_phi * ((2/3 * (1/2 * air_density * c2 * (phi_dot * (l6 + l7 - r3) + theta_dot * (l3 + l4 + (l6 + l7 - r3)*np.cos(phi)))**2) * (l6+l7-r3)) + (1/2 * air_density * c3 * (phi_dot * (l6+l7) + theta_dot * (l3 + l4 + (l6 + l7)*np.cos(phi)))**2) * (l6+l7))
    
    else:
        # Default (air_resistance_type = 0). Does not use air resistance in integration.
        air_res_theta_dot = 0
        air_res_phi_dot = 0

    return [air_res_theta_dot, air_res_phi_dot]


def friction_torque (theta_dot, phi_dot) -> list[float]:
    friction_theta: float
    friction_phi: float

    # Determine which direction the force should act depending on what direction the angular velocities have.
    if not theta_dot == 0:
        dir_theta = (-1) * theta_dot / np.abs(theta_dot)
    else:
        dir_theta = 0

    if not phi_dot == 0:
        dir_phi = (-1) * phi_dot / np.abs(phi_dot)
    else:
        dir_phi = 0

    # Determine which type of friction to use. Default (else) is 0 (no friciton)
    if friction_type == 1:
        friction_theta = dir_theta * fric * g * (m1 + m2 + m3)
        friction_phi   = dir_phi   * fric * g * (m2 + m3)
    else:
        friction_theta = 0.0
        friction_phi = 0.0

    return [friction_theta, friction_phi]


# F-matrix
# Unit forces: [N], Unit moments: [Nm]
def F (theta, phi, theta_dot, phi_dot) -> np.array:
    return np.array([
        [    0],
        [    0],
        [-m1*g],
        [air_resistance_torque(theta, phi, theta_dot, phi_dot)[0] + friction_torque(theta_dot, phi_dot)[0]], # M = (l3+l4) * F_air, F_air = (1/2 * c * air_density * A * v^2), v = theta_dot * (l3+l4)
        [    0],
        [    0],
        [    0],
        [    0],
        [-m2*g],
        [air_resistance_torque(theta, phi, theta_dot, phi_dot)[1] + friction_torque(theta_dot, phi_dot)[1]], # M = (l3+l4) * F_air, F_air = (1/2 * c * air_density * A * v^2), v = phi_dot * (l6+l7-r3)
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

# Time [s]
time: float

# Angles [rad]
theta: float
phi: float

# Angular velocity [rad/s]
theta_dot: float
phi_dot: float

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
    B_matrix = np.array([
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
    #print(B_matrix, end="\n\n")
    return B_matrix


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


def F_star (theta, phi, theta_dot, phi_dot) -> np.array:
    """
        F* = B_T * F
    """
    return np.dot( B_T(theta, phi), F(theta, phi, theta_dot, phi_dot) )


def Q (theta_dot, phi_dot):
    return np.array([
        [theta_dot],
        [phi_dot]])


def solve_Q_dot (time, theta, theta_dot, phi, phi_dot) -> np.array:
    """
        Equation of motion solving for Q_dot.
        Q_dot = inv(M*) * (F* - N* * Q)
    """
    return np.subtract( np.dot( M_star_inv(theta, phi), F_star(theta, phi, theta_dot, phi_dot) ), np.dot( N_star(theta, theta_dot, phi, phi_dot), Q(theta_dot, phi_dot) ) )


def integrate_Q_dot (time, data) -> np.array:
    

    return solve_Q_dot(theta=data[0], theta_dot=data[1], phi=data[2], phi_dot=[3])

