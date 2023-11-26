from Numerical_Integration import runge_kutta, euler_explicit, euler_implicit, linear_multistep

"""
    Run Runge-Kutta integrations
"""
# Test for 100s time, for dt = 0.1
#big_timestep = runge_kutta(time_end=100, timestep=0.1, title="BigTimeStep")

# Test for 100s time, for a moderately small timestep. No air resistance (default).
no_air_resistance = runge_kutta(time_end=100, timestep=0.01, title="NoAirResistance")

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
    Run Euler-Implicit
"""
#euler_implicit(time_end=100, timestep=0.01, title="NoAirResistance")

"""
    Run Linear Multi-step (Adam Bashforth) method integrations
"""
# Test for 100s time with timestep 0.01s, same as what we use on most Runge-Kutta integrations for comparison. No air resistance.
#linear_multistep(time_end=100, timestep=0.1, title="Timestep_0,1")

# Test for 100s time with timestep 0.01s
#linear_multistep(time_end=100, timestep=0.01, title="Timestep_0,01")

# Test for 100s time with timestep 0.0001s
#linear_multistep(time_end=100, timestep=0.001, title="Timestep_0,001")

# Test for timestep 0.1. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#linear_multistep(time_end=20, timestep=0.1, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,1")

# Test for timestep 0.01. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#linear_multistep(time_end=20, timestep=0.01, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,01")

# Test for timestep 0.001. Drops the arms from pointing 30 degrees upwards (theta = 1/6 * pi).
#linear_multistep(time_end=20, timestep=0.001, theta_start=1/6*np.pi, title="Drop30deg_timestep_0,001")

# Test for timestep 0.1. Simple air resistance.
#linear_multistep(time_end=100, timestep=0.1, title="SimpleAirResistance_timestep_0,1", air_resistance_type=1)

# Test for timestep 0.01. Simple air resistance.
#linear_multistep(time_end=100, timestep=0.01, title="SimpleAirResistance_timestep_0,01", air_resistance_type=1)

# Test for timestep 0.001. Simple air resistance.
#linear_multistep(time_end=100, timestep=0.001, title="SimpleAirResistance_timestep_0,001", air_resistance_type=1)