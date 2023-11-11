import matplotlib.pyplot as plt
import pandas as pd

def make_graph (filepath: str):
    """
        Plots the graphs from the specified file made by the numerical integration. Saves the graphs to a PNG file in the results folder.
    """
    dataframe = pd.read_csv(filepath)
    dataframe.head() # I have very little clue what this actually does other than selecting all the rows. All i know is it is required to use for pandas plotting with dataframes.

    # Define the figure for the plots and the figure size. Note that this size likely won't fit well on the screen, depending on how big the screen is.
    fig = plt.figure(figsize=(16,16))

    data_plots = dataframe.plot(x="time", y=["theta", "phi", "theta_dot", "phi_dot"], title=f"Data: {filepath.replace('results/', '')}.   Timestep (dt): {dataframe['timestep'][0]}", subplots=True, ax=fig)
    for dp in data_plots:
        dp.set_xlabel("Time [s]")
        dp.legend(loc="upper left")
    data_plots[0].set_ylabel("Angle [rad]")
    data_plots[1].set_ylabel("Angle [rad]")
    data_plots[2].set_ylabel("Angular velocity [rad/s]")
    data_plots[3].set_ylabel("Angular velocity [rad/s]")
    fig.savefig(f"results/Graph Figure {filepath.replace('results/', '').replace('.csv', '')}.png")
    

# Plot the test cases (Comment these out when not testing!)
make_graph("results/RK4_man_NoAirResist, time00,40,09 date12,11,2023.csv")
make_graph("results/RK4_man_NoAirResist, time00,40,15 date12,11,2023.csv")
make_graph("results/RK4_man_NoAirResist, time00,40,26 date12,11,2023.csv")
make_graph("results/RK4_man_NoAirResist, time00,40,38 date12,11,2023.csv")
make_graph("results/RK4_man_NoAirResist, time00,40,43 date12,11,2023.csv")

"""
# COMMENT / UNCOMMENT THE BELOW LINE DEPENDING ON IF YOU WANT TO SHOW THE GRAPH WHEN RUNNING THE PROGRAM. #
# NOTE THAT SOME OF THE AXES LOOK STRANGE WHILE RUNNING THE SCRIPT BUT LOOKS FINE IN THE PNG FILE LATER.  #
# HOWEVER TO TARGET AND GET INFO FORM A SPECIFIC POINT USING THE SCRIPT IS BETTER.                        #
"""
# plt.show()