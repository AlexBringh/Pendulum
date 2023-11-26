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
    
"""
# COMMENT / UNCOMMENT THE BELOW LINE DEPENDING ON IF YOU WANT TO SHOW THE GRAPH WHEN RUNNING THE PROGRAM. #
# NOTE THAT SOME OF THE AXES LOOK STRANGE WHILE RUNNING THE SCRIPT BUT LOOKS FINE IN THE PNG FILE LATER.  #
# HOWEVER TO TARGET AND GET INFO FORM A SPECIFIC POINT USING THE SCRIPT IS BETTER.                        #
"""

def compare_two_graphs (data1_path: str, data2_path: str, data1_title: str, data2_title: str):
    """
        Plots graphs together and compares
    """

    dataframe_1 = pd.read_csv(data1_path)
    dataframe_2 = pd.read_csv(data2_path)

    data1_time = dataframe_1["time"].to_numpy()
    data2_time = dataframe_2["time"].to_numpy()

    data1_theta = dataframe_1["theta"].to_numpy()
    data2_theta = dataframe_2["theta"].to_numpy()

    data1_theta_dot = dataframe_1["theta_dot"].to_numpy()
    data2_theta_dot = dataframe_2["theta_dot"].to_numpy()

    data1_phi = dataframe_1["phi"].to_numpy()
    data2_phi = dataframe_2["phi"].to_numpy()

    data1_phi_dot = dataframe_1["phi_dot"].to_numpy()
    data2_phi_dot = dataframe_2["phi_dot"].to_numpy()

    theta_compare = plt.figure(figsize=(16,16))
    plt.plot(data1_time, data1_theta, '-r', label=f"Theta: {data1_title}")
    plt.plot(data2_time, data2_theta, '--b', label=f"Theta: {data2_title}")
    plt.legend()

    phi_compare = plt.figure(figsize=(16,16))
    plt.plot(data1_time, data1_phi, '-g', label=f"Phi: {data1_title}")
    plt.plot(data2_time, data2_phi, '--m', label=f"Phi: {data2_title}")
    plt.legend()

    theta_dot_compare = plt.figure(figsize=(16,16))
    plt.plot(data1_time, data1_theta_dot, '-y', label=f"Theta_dot: {data1_title}")
    plt.plot(data2_time, data2_theta_dot, '--k', label=f"Theta_dot: {data2_title}")
    plt.legend()

    phi_dot_compare = plt.figure(figsize=(16,16))
    plt.plot(data1_time, data1_phi_dot, '-y', label=f"Phi_dot: {data1_title}")
    plt.plot(data2_time, data2_phi_dot, '--b', label=f"Phi_dot: {data2_title}")
    plt.legend()

    try:
        difference_fig, axs = plt.subplots(ncols=2, nrows=2)
        difference_fig.suptitle("Difference between values of the 2 datasets.")
        axs[0,0].plot(data1_time, (data1_theta-data2_theta), '-b')
        axs[0,0].text(0,0, s="Difference Theta")
        axs[0,1].plot(data1_time, (data1_phi-data2_phi), '-g')
        axs[0,1].text(0,0, s="Difference Phi")
        axs[1,0].plot(data1_time, (data1_theta_dot-data2_theta_dot), '-k')
        axs[1,0].text(0,0, s="Difference Theta_dot")
        axs[1,1].plot(data1_time, (data1_phi_dot-data2_phi_dot), '-o')
        axs[1,1].text(0,0, s="Difference Phi_dot")
    except:
        pass

    plt.legend()

    theta_compare.savefig(f"results/Compare/Graph Theta Compare {data1_title} and {data2_title}.png")
    phi_compare.savefig(f"results/Compare/Graph Phi Compare {data1_title} and {data2_title}.png")
    theta_dot_compare.savefig(f"results/Compare/Graph Theta_dot Compare {data1_title} and {data2_title}.png")
    phi_dot_compare.savefig(f"results/Compare/Graph Phi_dot Compare {data1_title} and {data2_title}.png")
    difference_fig.savefig(f"results/Compare/Graph Compare Differences {data1_title} and {data2_title}.png")
    #plt.show()


# plt.show()