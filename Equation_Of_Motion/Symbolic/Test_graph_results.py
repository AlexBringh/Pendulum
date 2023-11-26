from Graph_Results import compare_two_graphs

title1 = "RK4-Explicit"
path1 = "results/RK4_dt0,01_NoAirRes, time21,27,21 date26,11,2023.csv"

title2 = "Euler-implicit"
path2 = "results/EulerImp_dt0,01_NoAirRes, time21,27,38 date26,11,2023.csv"

compare_two_graphs(data1_path=path1, data1_title=title1, data2_path=path2, data2_title=title2)

