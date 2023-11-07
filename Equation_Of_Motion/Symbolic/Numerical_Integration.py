import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import Equation_Of_Motion as eom
import numpy as np

"""
    Runge Kutta numerical integration method
"""
<<<<<<< HEAD
def calculate_derivatives(Q, theta, theta_dot, phi, phi_dot):
    M_star_inv_matrix = eom.M_star_inv(theta, phi)
    N_star_matrix = eom.N_star(theta, theta_dot, phi, phi_dot)
    F_star_matrix = eom.F_star(theta, phi)
    Q_dot = np.dot(M_star_inv_matrix, (F_star_matrix - np.dot(N_star_matrix, Q)))
    return Q_dot

def runge_kutta (Q, theta, theta_dot, phi, phi_dot, dt):
    k1 = calculate_derivatives(Q, theta, theta_dot, phi, phi_dot)
    k2 = calculate_derivatives(Q + k1 * (dt / 2), theta, theta_dot, phi, phi_dot)
    k3 = calculate_derivatives(Q + k2 * (dt / 2), theta, theta_dot, phi, phi_dot)
    k4 = calculate_derivatives(Q + k3 * dt, theta, theta_dot, phi, phi_dot)

    Q_next = Q + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return Q_next
=======
>>>>>>> f34cc0a4ed671a2ea1bc247296c87693520f28ff
