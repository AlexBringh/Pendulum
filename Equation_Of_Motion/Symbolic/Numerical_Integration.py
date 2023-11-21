import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from datetime import datetime
from time import sleep

import Equation_Of_Motion as eom
import Graph_Results as graph


def runge_kutta (time_end, time_start:float = 0, timestep:float = 0.01, theta_start:float = 0, theta_dot_start:float = 0.0, phi_start:float = 0, phi_dot_start:float = 0, air_resistance_type: int = 0, friction_type: int = 0, decimal_roundoff: int = 10, title: str = "") -> pd.DataFrame:
    """
        Fourth order. Explicit Runge-Kutta method for numerical integration.
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
    eom.air_resistance_type = air_resistance_type # Set setting for air resistance or not.
    eom.friction_type = friction_type # Set setting for friction or not.

    # Define lists for storing time, theta, phi, theta_dot, phi_dot, theta_dot_dot and phi_dot_dot
    store_time: list = []
    store_theta: list = []
    store_phi: list = []
    store_theta_dot: list = []
    store_phi_dot: list = []

    # Store initial values to the results.
    store_time.append(eom.time)
    store_theta.append(eom.theta)
    store_phi.append(eom.phi)
    store_theta_dot.append(eom.theta_dot)
    store_phi_dot.append(eom.phi_dot)
    
    print(f"\nRunning Runge-Kutta 4th order for: {title}, time-target:{time_end}, timestep:{dt}, decimal_roundoff:{dro}, theta0:{theta_start}, phi0:{phi_start}, theta_dot0:{theta_dot_start}, phi_dot0:{phi_dot_start}")

    while eom.time < time_end:
        print(f"\rTime: {eom.time}, theta: {round(eom.theta, dro)}, phi: {round(eom.phi, dro)}", end="")
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

    print("")

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

    save_results(results=results_list, title=(f"RK4_{title}"))

    return results_list


def euler_explicit (time_end, time_start:float = 0, timestep:float = 0.01, theta_start:float = 0, theta_dot_start:float = 0.0, phi_start:float = 0, phi_dot_start:float = 0, air_resistance_type: int = 0, friction_type: int = 0, decimal_roundoff: int = 10, title: str = "") -> pd.DataFrame:
    """
        Explicit Euler method for numerical integration.
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
    eom.air_resistance_type = air_resistance_type # Set setting for air resistance or not.
    eom.friction_type = friction_type # Set setting for friction or not.

    # Define lists for storing time, theta, phi, theta_dot, phi_dot, theta_dot_dot and phi_dot_dot
    store_time: list = []
    store_theta: list = []
    store_phi: list = []
    store_theta_dot: list = []
    store_phi_dot: list = []

    # Store initial values to the results.
    store_time.append(eom.time)
    store_theta.append(eom.theta)
    store_phi.append(eom.phi)
    store_theta_dot.append(eom.theta_dot)
    store_phi_dot.append(eom.phi_dot)
    
    print(f"\nRunning Explicit-Euler for: {title}, time-target:{time_end}, timestep:{dt}, decimal_roundoff:{dro}, theta0:{theta_start}, phi0:{phi_start}, theta_dot0:{theta_dot_start}, phi_dot0:{phi_dot_start}")

    while eom.time < time_end:
        print(f"\rTime: {eom.time}, theta: {round(eom.theta, dro)}, phi: {round(eom.phi, dro)}", end="")
        
        Q_new = dt * eom.solve_Q_dot(time=eom.time, theta=eom.theta, phi=eom.phi, theta_dot=eom.theta_dot, phi_dot=eom.phi_dot)

        eom.theta_dot = float(Q_new[0]) + eom.theta_dot
        eom.phi_dot = float(Q_new[1]) + eom.phi_dot

        eom.theta = eom.theta + eom.theta_dot * dt
        eom.phi = eom.phi + eom.phi_dot * dt

        eom.time = eom.time + dt

        eom.time = round(eom.time, dro)
        eom.theta = round(eom.theta, dro)
        eom.phi = round(eom.phi, dro)
        eom.theta_dot = round(eom.theta_dot, dro)
        eom.phi_dot = round(eom.phi_dot, dro)
    
        store_time.append(eom.time)
        store_theta.append(eom.theta)
        store_phi.append(eom.phi)
        store_theta_dot.append(eom.theta_dot)
        store_phi_dot.append(eom.phi_dot)

    print("")

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

    save_results(results=results_list, title=(f"EulerExp_{title}"))

    return results_list


def linear_multistep (time_end, time_start:float = 0, timestep:float = 0.01, theta_start:float = 0, theta_dot_start:float = 0.0, phi_start:float = 0, phi_dot_start:float = 0, air_resistance_type: int = 0, friction_type: int = 0, decimal_roundoff: int = 10, title: str = "") -> pd.DataFrame:
    """
        Explicit LInear-Multistep (Adam-Bashforth) method for numerical integration.
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
    eom.air_resistance_type = air_resistance_type # Set setting for air resistance or not.
    eom.friction_type = friction_type # Set setting for friction or not.

    # Define lists for storing time, theta, phi, theta_dot, phi_dot, theta_dot_dot and phi_dot_dot
    store_time: list = []
    store_theta: list = []
    store_phi: list = []
    store_theta_dot: list = []
    store_phi_dot: list = []

    # Store initial values to the results.
    store_time.append(eom.time)
    store_theta.append(eom.theta)
    store_phi.append(eom.phi)
    store_theta_dot.append(eom.theta_dot)
    store_phi_dot.append(eom.phi_dot)
    
    print(f"\nRunning Explicit-Linear-Multistep (Adam-Bashforth method), two-step for: {title}, time-target:{time_end}, timestep:{dt}, decimal_roundoff:{dro}, theta0:{theta_start}, phi0:{phi_start}, theta_dot0:{theta_dot_start}, phi_dot0:{phi_dot_start}")

    # For two-step, we must first solve the first approximation using Euler's method.
    print(f"\rTime: {eom.time}, theta: {round(eom.theta, dro)}, phi: {round(eom.phi, dro)}", end="")
        
    Q_new = dt * eom.solve_Q_dot(time=eom.time, theta=eom.theta, phi=eom.phi, theta_dot=eom.theta_dot, phi_dot=eom.phi_dot)

    eom.theta_dot = float(Q_new[0]) + eom.theta_dot
    eom.phi_dot = float(Q_new[1]) + eom.phi_dot

    eom.theta = eom.theta + eom.theta_dot * dt
    eom.phi = eom.phi + eom.phi_dot * dt

    eom.time = eom.time + dt

    eom.time = round(eom.time, dro)
    eom.theta = round(eom.theta, dro)
    eom.phi = round(eom.phi, dro)
    eom.theta_dot = round(eom.theta_dot, dro)
    eom.phi_dot = round(eom.phi_dot, dro)
    
    store_time.append(eom.time)
    store_theta.append(eom.theta)
    store_phi.append(eom.phi)
    store_theta_dot.append(eom.theta_dot)
    store_phi_dot.append(eom.phi_dot)

    i: int = 0 # iterator

    while eom.time < time_end:
        print(f"\rTime: {eom.time}, theta: {round(eom.theta, dro)}, phi: {round(eom.phi, dro)}", end="")
        Q_new = dt * (3/2 * eom.solve_Q_dot(time=eom.time, theta=eom.theta, phi=eom.phi, theta_dot=eom.theta_dot, phi_dot=eom.phi_dot) - 1/2 * eom.solve_Q_dot(time=store_time[i], theta=store_theta[i], phi=store_phi[i], theta_dot=store_theta_dot[i], phi_dot=store_phi_dot[i]))
        
        eom.theta_dot = float(Q_new[0]) + eom.theta_dot
        eom.phi_dot = float(Q_new[1]) + eom.phi_dot

        eom.theta = eom.theta + eom.theta_dot * dt
        eom.phi = eom.phi + eom.phi_dot * dt

        eom.time = eom.time + dt

        eom.time = round(eom.time, dro)
        eom.theta = round(eom.theta, dro)
        eom.phi = round(eom.phi, dro)
        eom.theta_dot = round(eom.theta_dot, dro)
        eom.phi_dot = round(eom.phi_dot, dro)
    
        store_time.append(eom.time)
        store_theta.append(eom.theta)
        store_phi.append(eom.phi)
        store_theta_dot.append(eom.theta_dot)
        store_phi_dot.append(eom.phi_dot)

        i = i + 1

    print("")

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

    save_results(results=results_list, title=(f"LinearTwostep_{title}"))

    return results_list


def save_results (results:pd.DataFrame, title:str):
    """
        Save results from numerical integrations
    """ 
    current_datetime = datetime.now()
    try: 
        results.to_csv(f"results/{title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv")
        print(f"Succesfully wrote results to file: {title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv!")
        graph.make_graph(f"results/{title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv")

    except:
        print(f"Error. Failed to write results to file: {title}, time{current_datetime.strftime('%H,%M,%S')} date{current_datetime.strftime('%d,%m,%Y')}.csv!")
    sleep(5) # Just a delay so that the files won't overwrite each other in case there are more than just 1.
    

"""
    Run Runge-Kutta integrations
"""
# Test for 100s time, for dt = 0.1
#big_timestep = runge_kutta(time_end=100, timestep=0.1, title="BigTimeStep")

# Test for 100s time, for a moderately small timestep. No air resistance (default).
#no_air_resistance = runge_kutta(time_end=100, timestep=0.01, title="NoAirResistance")

# Test for the same as above, but include the simple air resistance model now.
#simple_air_resistance = runge_kutta(time_end=100, timestep=0.01, air_resistance_type=1, title="SimpleAirResistance")

# Test again for the same as above, but this time include the complex air resistance model.
#complex_air_resistance = runge_kutta(time_end=100, timestep=0.01, air_resistance_type=2, title="ComplexAirResistance")

# Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#drop_30_degree_up = runge_kutta(time_end=20, timestep=0.01, theta_start=1/6*np.pi, title="DropFrom30DegreesUp")

# Gives the arms an initial velocity
#initial_velocity = runge_kutta(time_end=1000, timestep=0.01, title="InitialVelocity", theta_dot_start=-0.5, phi_dot_start=-1.5)


"""
    Run Explicit Euler integrations
"""
# Test for 100s time with timestep 0.01s, same as what we use on most Runge-Kutta integrations for comparison. No air resistance.
#euler_explicit(time_end=100, timestep=0.1, title="Timestep_0,1")

# Test for 100s time with timestep 0.01s
#euler_explicit(time_end=100, timestep=0.01, title="Timestep_0,01")

# Test for 100s time with timestep 0.0001s
#euler_explicit(time_end=100, timestep=0.001, title="Timestep_0,001")

# Test for timestep 0.1. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#euler_explicit(time_end=20, timestep=0.1, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,1")

# Test for timestep 0.01. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#euler_explicit(time_end=20, timestep=0.01, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,01")

# Test for timestep 0.001. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#euler_explicit(time_end=20, timestep=0.001, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,001")

# Test for timestep 0.1. Simple air resistance.
#euler_explicit(time_end=100, timestep=0.1, title="SimpleAirResistance_timestep_0,1", air_resistance_type=1)

# Test for timestep 0.01. Simple air resistance.
#euler_explicit(time_end=100, timestep=0.01, title="SimpleAirResistance_timestep_0,01", air_resistance_type=1)

# Test for timestep 0.001. Simple air resistance.
#euler_explicit(time_end=100, timestep=0.001, title="SimpleAirResistance_timestep_0,001", air_resistance_type=1)


"""
    Run Linear Multi-step (Adam Bashforth) method integrations
"""
# Test for 100s time with timestep 0.01s, same as what we use on most Runge-Kutta integrations for comparison. No air resistance.
#linear_multistep(time_end=100, timestep=0.1, title="Timestep_0,1")

# Test for 100s time with timestep 0.01s
#linear_multistep(time_end=100, timestep=0.01, title="Timestep_0,01")

# Test for 100s time with timestep 0.0001s
linear_multistep(time_end=100, timestep=0.001, title="Timestep_0,001")

# Test for timestep 0.1. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
linear_multistep(time_end=20, timestep=0.1, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,1")

# Test for timestep 0.01. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
linear_multistep(time_end=20, timestep=0.01, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,01")

# Test for timestep 0.001. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
linear_multistep(time_end=20, timestep=0.001, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,001")

# Test for timestep 0.1. Simple air resistance.
linear_multistep(time_end=100, timestep=0.1, title="SimpleAirResistance_timestep_0,1", air_resistance_type=1)

# Test for timestep 0.01. Simple air resistance.
linear_multistep(time_end=100, timestep=0.01, title="SimpleAirResistance_timestep_0,01", air_resistance_type=1)

# Test for timestep 0.001. Simple air resistance.
linear_multistep(time_end=100, timestep=0.001, title="SimpleAirResistance_timestep_0,001", air_resistance_type=1)

print("\nNumerical integration has finished.")