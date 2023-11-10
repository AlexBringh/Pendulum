import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv
from datetime import datetime

import Equation_Of_Motion as eom


def runge_kutta (time_start, time_end, dt: float = 0.1, theta_start: float = 0, theta_dot_start: float = 0.0, theta_dot_dot_start: float = 0, phi_start: float = 0, phi_dot_start: float = 0, phi_dot_dot_start: float = 0) -> list[list]:
    """
        Fourth order. explicit Runge-Kutta method for numerical integration.
        Can specify starting points for:
        time_start, time_end, (time) 
        dt (timestep)
        theta, phi (angles)
        theta_dot, phi_dot (angular velocity)
        theta_dot_dot, phi_dot_dot (angular acceleration)
    """
    dt: float # timestep
    # Set starting-points.
    eom.time = time_start
    eom.theta = theta_start
    eom.phi = phi_start
    eom.theta_dot = theta_dot_start
    eom.phi_dot = phi_dot_start

    # Define lists for storing time, theta, phi, theta_dot, phi_dot, theta_dot_dot and phi_dot_dot
    store_time: list = []
    store_theta: list = []
    store_phi: list = []
    store_theta_dot: list = []
    store_phi_dot: list = []
    store_theta_dot_dot: list = []
    store_phi_dot_dot: list = []
    
    while eom.time < time_end:

        # Calculate k - corrector values from the function for the differential equation, 'solve_Q_dot'
        k1: np.array = eom.solve_Q_dot(time=eom.time, theta=eom.theta, theta_dot=eom.theta_dot, phi=eom.phi, phi_dot=eom.phi_dot)
        k2: np.array = eom.solve_Q_dot(time=eom.time + dt/2, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k1[0])/2, phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k1[1]) / 2)
        k3: np.array = eom.solve_Q_dot(time=eom.time + dt/2, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k2[0])/2, phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k2[1]) / 2)
        k4: np.array = eom.solve_Q_dot(time=eom.time + dt, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k3[0]), phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k3[1]))

        # Add to Q_dot (acceleration terms, also known as q_dot_dot)
        eom.theta_dot_dot = eom.theta_dot_dot + 1/6 * (k1[0] + 2*k2[0] + 2*k3[3] + k4[0])
        eom.phi_dot_dot = eom.phi_dot_dot + 1/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        # Approximate new Q (velocity terms, also known as q_dot)
        eom.theta_dot = eom.theta_dot + eom.theta_dot_dot * dt
        eom.phi_dot = eom.phi_dot + eom.phi_dot_dot * dt

        # Approximate new q (angle terms)
        eom.theta = eom.theta + eom.theta_dot * dt
        eom.phi = eom.phi + eom.phi_dot * dt

        # Update time
        eom.time = eom.time + dt

        # Save current values.
        store_time.append(eom.time)
        store_theta.append(eom.theta)
        store_phi.append(eom.phi)
        store_theta_dot.append(eom.theta_dot)
        store_phi_dot.append(eom.phi_dot)
        store_theta_dot_dot.append(eom.theta_dot_dot)
        store_phi_dot_dot.append(eom.phi_dot_dot)

    # Define return list
    results_list: list = [
        store_time,
        store_theta,
        store_phi,
        store_theta_dot,
        store_phi_dot,
        store_theta_dot_dot,
        store_phi_dot_dot
    ]

    current_datetime = datetime.now()
    save_results(results=results_list, timestep=dt, title=(f"Runge_Kutta_integration, {current_datetime.strftime('%H,%M,%S %d,%m,%Y')}"))

    return results_list



def save_results (results: list[list], timestep: float, title: str):
    """
        Save results
    """
    pass