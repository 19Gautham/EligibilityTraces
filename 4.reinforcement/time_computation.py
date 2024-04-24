from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util
import layout
import sys
import types
import time
import random
import os
import subprocess

from scipy.stats import shapiro
from scipy.stats import ttest_rel


base_dir_path = "C:\\Users\\gdjk9\\OneDrive\\Desktop\\CSE 571 Artifical Intelligence\\Project\\Project 3\\3.reinforcement\\4.reinforcement\\"
eval_layout_path = "C:\\Users\\gdjk9\\OneDrive\\Desktop\\CSE 571 Artifical Intelligence\\Project\\Project 3\\3.reinforcement\\4.reinforcement\\layouts"
result_path = "C:\\Users\\gdjk9\\OneDrive\\Desktop\\CSE 571 Artifical Intelligence\\Project\\Project 3\\3.reinforcement\\4.reinforcement\\Results"

def sortkey(filename):
    # Extract the number part of the filename and convert it to integer
    # print(filename.split('_'))
    return int(filename.split('_')[1].split('.')[0])


layouts = []

# List all files in the directory
files = os.listdir(eval_layout_path)
files = sorted([file for file in files if file.startswith("test")], key=sortkey)

print("File list: {}".format(files))
print("semi-gradient td lambda")

ignore_list = []
# Semi gradient agent
# for file in files:
#     if file.startswith("test_"):
#         layout_name = file.split('.')[0]
#         if int(layout_name.split('_')[1]) in ignore_list: 
#             continue
#         print("\nLayout: {}".format(layout_name))
#
#         subprocess.run(["python", "pacman.py", "-p", "SemiGradientTDAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])
#
# os.rename(base_dir_path + "score_0.txt", base_dir_path + "score_semi_gradient_agent.txt")

with open(base_dir_path + "training_time_SGTD.txt", "r") as file:
    training_time_SGTD = [float(line.strip()) for line in file]

print("End of semi-gradient td lambda")

# ignore_list = [1, 5, 11, 25, 26, 35, 52, 62, 64, 68, 72, 78, 81, 92, 97, 99]

print("true online td lambda")
#
#
# True online TD lambda agent
# for file in files:
#     if file.startswith("test_"):
#         # Extract layout name from file
#         layout_name = file.split('.')[0]
#         # if int(layout_name.split('_')[1]) in ignore_list: 
#         #     continue
#         print("\nLayout: {}".format(layout_name))
#         # Run the command
#         subprocess.run(["python", "pacman.py", "-p", "TrueOnlineTdLambdaAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])
#
# os.rename(base_dir_path + "score_0.txt", base_dir_path + "score_true_online_td_lambda_agent.txt")
#
with open(base_dir_path + "training_time_TOTD.txt", "r") as file:
    training_time_TOTD = [float(line.strip()) for line in file]

print("End of true online td lambda")


# for file in files:
#     if file.startswith("test_"):
#         # Extract layout name from file
#         layout_name = file.split('.')[0]
#         if int(layout_name.split('_')[1]) in ignore_list: 
#             continue
#         # Run the command
#         subprocess.run(["python", "pacman.py", "-p", "ApproximateQAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])
#
# os.rename(base_dir_path + "score_0.txt", base_dir_path + "score_q_learning_agent.txt")

# with open(base_dir_path + "training_time_AQA.txt", "r") as file:
#     training_time_AQA = [float(line.strip()) for line in file]

# scores_q_learning = [1821.2, -163.26666666666668, 1599.4, 1284.0, 1483.6666666666667, 786.8666666666667, 1080.0, 617.4, 606.4, 1000.1333333333333, 2334.8, 750.6, 1038.8666666666666, 959.2, 378.93333333333334, 2159.8, 1865.6, 689.1333333333333, 1247.0666666666666, 2567.0666666666666, 656.1333333333333, 2619.6666666666665, 785.2666666666667, 2059.2, 366.2, 1345.9333333333334, 307.26666666666665, 170.93333333333334, 1283.4, 838.6, 924.5333333333333, 574.8, 1316.9333333333334, 605.5333333333333, -129.6, 1972.3333333333333, 804.8666666666667, 1295.8, 826.8666666666667, 269.26666666666665, 783.3333333333334, 48.13333333333333, 2401.2, 733.6, 526.8, 439.3333333333333, 627.0666666666667, 1294.7333333333333, 962.1333333333333, 504.26666666666665, 1007.5333333333333, 389.73333333333335, 484.46666666666664, 1660.5333333333333, 1128.6666666666667, 793.0, 1437.6, 706.6, 453.93333333333334, 804.6, 1435.6, 6.866666666666666, 925.2666666666667, 3135.733333333333, 506.73333333333335, 2312.266666666667, 201.8, 1248.0, 2112.6666666666665, 587.8666666666667, 1875.0666666666666, 328.3333333333333, -248.2, 985.7333333333333, 664.8666666666667, 512.9333333333333, 2259.8, 498.1333333333333, -56.333333333333336, 342.0, 862.9333333333333, 1886.0666666666666, 1371.8666666666666, 1781.9333333333334, 822.6666666666666, 1171.2666666666667, 470.2, 221.8, 33.666666666666664, 1296.4, 1798.8, 748.8666666666667, 297.8, 789.2, 1890.6, 846.8, 1046.3333333333333, 734.4, 700.2666666666667, 699.0666666666667]

# q_score = scores_q_learning
# for i in range(len(training_time_TOTD)):
#     if i in ignore_list:
#         continue
#
#     q_score.append(training_time_TOTD[i])
#
print("Scores (Semi-Gradient): {}".format(training_time_SGTD))
print("Scores (True-Online TD Lambda): {}".format(len(training_time_TOTD)))
# print("Scores (Approximate Q Agent): {}".format(len(training_time_AQA)))
#
# l1 = [10, 15, 19, 21, 42, 63, 65, 68, 76]
# l2 =  [21, 22, 31, 45, 53, 55, 58, 62, 66]
#
# union_list = list(set(l1 + l2))
#
z = []
# filtered_totd = []
# training_time_SGTD = [670.6, 1717.8, 205.33333333333334, 944.8666666666667, 445.46666666666664, 676.1333333333333, 531.4, 863.9333333333333, 2943.866666666667, 745.0666666666667, 1490.0, 839.8666666666667, 207.0, 1928.5333333333333, 1251.1333333333334, 510.8, 1343.0, 1157.8, 863.4666666666667, 1968.1333333333334, 1574.4, 730.5333333333333, 450.8666666666667, 923.9333333333333, 975.2666666666667, 939.9333333333333, 741.7333333333333, 288.6666666666667, 1216.9333333333334, 915.0, 506.6, 2674.5333333333333, 2022.8666666666666, 594.0666666666667, 1114.9333333333334, 359.4, 588.3333333333334, 1057.8, 1297.8666666666666, 1409.8, 1859.3333333333333, -272.8666666666667, 1001.8, 639.7333333333333, 1264.4, 893.8, -28.933333333333334, 568.7333333333333, 1256.4, 1093.9333333333334, 451.0, 507.0, 764.2, 439.3333333333333, -192.6, 189.46666666666667, 2890.5333333333333, 1286.9333333333334, 1852.3333333333333, 739.0, 758.0666666666667, 456.8666666666667, 249.33333333333334, 1057.4666666666667, 520.8, 2227.0, 546.6, 2415.266666666667, 782.8, 1207.0666666666666, 1803.5333333333333, 1642.6666666666667, 1295.6, 606.4, 2086.6, -206.73333333333332, 277.8, 613.6666666666666, 278.73333333333335, 843.2666666666667, 716.3333333333334, 1162.2, 1595.5333333333333, 1346.2, 318.1333333333333, 297.2, 269.3333333333333, 492.46666666666664, 742.3333333333334, 1192.8666666666666, 1063.2666666666667, 1145.0666666666666, 2471.3333333333335, 857.7333333333333, 314.8666666666667, 129.53333333333333, 469.3333333333333, 1837.9333333333334, 651.0, 1927.6]

# training_time_SGTD = [701.3333333333334, 1272.6666666666667, -96.66666666666667, 1063.4, 707.8666666666667, 440.06666666666666, 709.7333333333333, 869.9333333333333, 2239.4, 714.8, 1709.6666666666667, 931.0666666666667, 164.93333333333334, 2540.2, 786.4, 966.3333333333334, 1036.2666666666667, 819.7333333333333, 1019.0666666666667, 1507.0666666666666, 1867.4, 939.6666666666666, 531.0, 1184.2666666666667, 878.2666666666667, 1055.0, 616.0666666666667, 227.0, 991.3333333333334, 1026.4666666666667, 555.2, 2294.5333333333333, 1845.2666666666667, 551.6, 1324.9333333333334, 450.26666666666665, 808.6, 1054.5333333333333, 1200.8666666666666, 1469.8666666666666, 1932.1333333333334, -233.46666666666667, 1072.7333333333333, 963.4, 1340.3333333333333, 1050.1333333333334, 723.7333333333333, 576.6666666666666, 1091.6666666666667, 1199.8, 479.73333333333335, 309.73333333333335, 704.8, 791.2, 241.93333333333334, 404.46666666666664, 2826.6666666666665, 1393.6, 1773.9333333333334, 749.2666666666667, 865.6, 610.2, 71.2, 1007.2666666666667, 668.5333333333333, 2261.0666666666666, 146.53333333333333, 2677.0666666666666, 512.8666666666667, 1182.9333333333334, 1895.6, 1320.9333333333334, 1078.4, 834.0, 2166.8, -192.93333333333334, -17.866666666666667, 387.1333333333333, 227.2, 977.0, 1243.5333333333333, 1511.0, 1851.8666666666666, 1470.8, 592.2666666666667, 205.66666666666666, 492.93333333333334, 310.26666666666665, 890.5333333333333, 1555.1333333333334, 1474.0666666666666, 1140.2, 1852.4666666666667, 680.0666666666667, 598.0, -4.466666666666667, 489.4, 1962.8, 692.5333333333333, 1954.0666666666666]

# removed_list = []
for i in range(len(training_time_TOTD)):
    # if 0<= training_time_TOTD[i] <=2500:
    #     z.append(training_time_TOTD[i] - q_score[i])
    # else:
    #     removed_list.append(i)
    # if i in union_list:
    #     continue
    z.append(training_time_SGTD[i] - training_time_TOTD[i]) 
    # filtered_totd.append(training_time_TOTD[i])
    # filtered_q.append(q_score[i]) 

#######################################################################################

# print("Removed list: {}".format(removed_list))

# print("Filtered totd: {}".format(filtered_totd))
# print("Filtered q score: {}".format(filtered_q))
##############################################################################

print(z)

x = [1 if z[i] > 0 else 0 for i in range(len(z))]
c0 = 0
c1 = 0
for i in range(len(x)):
    if x[i]:
        c1 += 1
    else:
        c0 += 1
print("CO: {}".format(c0))
print("C1: {}".format(c1))


stat, p = shapiro(z)

print("Shapiro-Wilk Test Statistic:", stat)
print("P-value:", p)

if p > 0.05:
    print("The data appears to be normally distributed (fail to reject H0)")
else:
    print("The data does not appear to be normally distributed (reject H0)")


t_statistic, p_value = ttest_rel(training_time_TOTD, training_time_SGTD)

print("\n\nT-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("There is a statistically significant difference between the means.")
else:
    print("There is no statistically significant difference between the means.")

