import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from datetime import datetime
from time import sleep

import Equation_Of_Motion as eom


def runge_kutta (time_end, time_start:float = 0, timestep:float = 0.1, theta_start:float = 0, theta_dot_start:float = 0.0, phi_start:float = 0, phi_dot_start:float = 0, decimal_roundoff: int = 10) -> pd.DataFrame:
    """
        Fourth order. explicit Runge-Kutta method for numerical integration.
        Can specify starting points for:
        time_start, time_end, (time) 
        dt (timestep)
        theta, phi (angles)
        theta_dot, phi_dot (angular velocity)
        theta_dot_dot, phi_dot_dot (angular acceleration)
    """
    dt: float = timestep # timestep
    dro: int = decimal_roundoff # decimal round off to minimize the mess of small floating points
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
    
    while eom.time < time_end:
        # Calculate k - corrector values from the function for the differential equation, 'solve_Q_dot'
        k1: np.array = eom.solve_Q_dot(time=eom.time, theta=eom.theta, theta_dot=eom.theta_dot, phi=eom.phi, phi_dot=eom.phi_dot)
        k2: np.array = eom.solve_Q_dot(time=eom.time + dt/2, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k1[0])/2, phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k1[1]) / 2)
        k3: np.array = eom.solve_Q_dot(time=eom.time + dt/2, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k2[0])/2, phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k2[1]) / 2)
        k4: np.array = eom.solve_Q_dot(time=eom.time + dt, theta=eom.theta, theta_dot=eom.theta_dot + dt * float(k3[0]), phi=eom.phi, phi_dot=eom.phi_dot + dt * float(k3[1]))

        # Approximate new Q (velocity terms, also known as q_dot)
        eom.theta_dot = eom.theta_dot + 1/6 * dt * (round(float(k1[0]), dro) + 2*round(float(k2[0]), dro) + 2*round(float(k3[0]), dro) + round(float(k4[0]), dro))
        eom.phi_dot = eom.phi_dot + 1/6 * dt * (round(float(k1[1]), dro) + 2*round(float(k2[1]), dro) + 2*round(float(k3[1]), dro) + round(float(k4[1]), dro))

        # Approximate new q (angle terms)
        eom.theta = eom.theta + eom.theta_dot * dt
        eom.phi = eom.phi + eom.phi_dot * dt

        # Update time
        eom.time = round(eom.time + dt, dro)

        # Save current values.
        store_time.append(eom.time)
        store_theta.append(eom.theta)
        store_phi.append(eom.phi)
        store_theta_dot.append(eom.theta_dot)
        store_phi_dot.append(eom.phi_dot)

    # Define return list
    results_list: dict = {
        "time": store_time,
        "theta": store_theta,
        "phi": store_phi,
        "theta_dot": store_theta_dot,
        "phi_dot": store_phi_dot,
        "timestep": dt
    }
    results_list = pd.DataFrame(results_list)

    
    save_results(results=results_list, title=(f"RK4_man_NoAirResist"))

    return results_list


def save_results (results:pd.DataFrame, title:str):
    """
        Save results from numerical integrations
    """ 
    current_datetime = datetime.now()
    try: 
        results.to_csv(f"results/{title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv")
        print(f"Succesfully wrote results to file: {title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv!")
    except:
        print(f"Error. Failed to write results to file: {title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv!")
    sleep(5) # Just a delay so that the files won't overwrite each other in case there are more than just 1.
    

# The arms of the pendulum face straight down with these initial conditions. Should be NO development in movement. (apart from roundoff errors for floating point numbers) This one has a large timestep because the integration is simple.
test_case1 = runge_kutta(time_end=5, timestep=0.1, theta_start=-np.pi / 2, phi_start=0)

# The arms face straight out and drop down immediately after start. Larger timestep and longer duration. (The base position for theta and phi points straight out so no change is necessary.)
test_case2 = runge_kutta(time_end=10, timestep=0.01)

# A test where the timestep is much smaller. Same initial conditions and time target as the previous test case.
test_case3 = runge_kutta(time_end=10, timestep=0.001)

# Test for a  lang target time, for a moderately large timestep. Again standard initial conditions.
test_case4 = runge_kutta(time_end=100, timestep=0.01)

# The last test case drops from the arms pointing 30 degrees upwards (theta = 1/6 * pi).
test_case5 = runge_kutta(time_end=20, timestep=0.1, theta_start=1/6*np.pi)

print("Numerical integration has finished.")