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
files = sorted([file for file in files if file.startswith("test_")], key=sortkey)

print("File list: {}".format(files))
print("semi-gradient td lambda")

ignore_list = []
# # Semi gradient agent
# for file in files:
#     if file.startswith("test_"):
#         layout_name = file.split('.')[0]
#         if int(layout_name.split('_')[1]) in ignore_list: 
#             continue
#         print("\nLayout: {}".format(layout_name))
#
#         subprocess.run(["python", "pacman2.py", "-p", "SemiGradientTDAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])
#
# os.rename(base_dir_path + "score_1.txt", base_dir_path + "score_semi_gradient_agent.txt")
# with open(base_dir_path + "score_semi_gradient_agent.txt", "r") as file:
#     scores_semi_gradient = [float(line.strip()) for line in file]

print("End of semi-gradient td lambda")

# ignore_list = [1, 5, 11, 25, 26, 35, 52, 62, 64, 68, 72, 78, 81, 92, 97, 99]

print("true online td lambda")


# True online TD lambda agent
# for file in files:
#     if file.startswith("test_"):
#         # Extract layout name from file
#         layout_name = file.split('.')[0]
#         # if int(layout_name.split('_')[1]) in ignore_list: 
#         #     continue
#         print("\nLayout: {}".format(layout_name))
#         # Run the command
#         subprocess.run(["python", "pacman2.py", "-p", "TrueOnlineTdLambdaAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])
#
# os.rename(base_dir_path + "score_2.txt", base_dir_path + "score_true_online_td_lambda_agent.txt")

# with open(base_dir_path + "score_true_online_td_lambda_agent.txt", "r") as file:
#     scores_true_online_td_lambda = [float(line.strip()) for line in file]

print("End of true online td lambda")

for file in files:
    if file.startswith("test_"):
        # Extract layout name from file
        layout_name = file.split('.')[0]
        print("\nLayout: {}".format(layout_name))
        # Run the command
        subprocess.run(["python", "pacman.py", "-p", "ApproximateQAgent", "-x", "50", "-n", "65", "-l", layout_name, "-q"])

os.rename(base_dir_path + "score_0.txt", base_dir_path + "score_q_learning_agent.txt")
# #
with open(base_dir_path + "score_q_learning_agent.txt", "r") as file:
    scores_q_learning = [float(line.strip()) for line in file]

# with open(base_dir_path + "score_true_online_td_lambda_agent.txt", "r") as file:
#     scores_true_online_td_lambda = [float(line.strip()) for line in file]


# q_score = scores_q_learning
# for i in range(len(scores_true_online_td_lambda)):
#     if i in ignore_list:
#         continue
#
#     q_score.append(scores_true_online_td_lambda[i])
#
# print("Scores (Semi-Gradient): {}".format(scores_semi_gradient))
# print("Scores (True-Online TD Lambda): {}".format(len(scores_true_online_td_lambda)))
# print("Scores (Approximate Q Agent): {}".format(len(scores_q_learning)))
#
# z = []
# for i in range(len(scores_q_learning)):
#     z.append(scores_true_online_td_lambda[i] - scores_q_learning[i]) 
#
# print(z)
#
# # x = [1 if z[i] > 0 else 0 for i in range(len(z))]
# c0 = 0
# c1 = 0
# for i in range(len(z)):
#     if scores_true_online_td_lambda[i] > scores_q_learning[i]:
#         c1 += 1
#     else:
#         c0 += 1
# print("CO: {}".format(c0))
# print("C1: {}".format(c1))
#
#
# stat, p = shapiro(z)
#
# print("Shapiro-Wilk Test Statistic:", stat)
# print("P-value:", p)
#
# if p > 0.05:
#     print("The data appears to be normally distributed (fail to reject H0)")
# else:
#     print("The data does not appear to be normally distributed (reject H0)")
#
#
# t_statistic, p_value = ttest_rel(scores_q_learning, scores_true_online_td_lambda)
#
# print("\n\nT-statistic:", t_statistic)
# print("P-value:", p_value)
#
# if p_value < 0.05:
#     print("There is a statistically significant difference between the means.")
# else:
#     print("There is no statistically significant difference between the means.")

