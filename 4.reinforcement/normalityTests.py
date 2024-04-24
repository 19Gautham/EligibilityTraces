
from scipy.stats import shapiro
from scipy.stats import ttest_rel

base_dir_path = "C:\\Users\\gdjk9\\OneDrive\\Desktop\\CSE 571 Artifical Intelligence\\Project\\Project 3\\3.reinforcement\\4.reinforcement\\"

scores_AQA = []
scores_SGTD = []
scores_TOTD = []

with open(base_dir_path + "score_q_learning_agent.txt", "r") as file:
    scores_AQA = [float(line.strip()) for line in file]

# with open(base_dir_path + "score_semi_gradient_agent.txt", "r") as file:
#     scores_SGTD = [float(line.strip()) for line in file]

with open(base_dir_path + "score_true_online_td_lambda_agent.txt", "r") as file:
    scores_TOTD = [float(line.strip()) for line in file]

z = []

for i in range(len(scores_TOTD)):
    z.append(scores_TOTD[i] - scores_AQA[i]) 

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


t_statistic, p_value = ttest_rel(scores_TOTD, scores_AQA)

print("\n\nT-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("There is a statistically significant difference between the means.")
else:
    print("There is no statistically significant difference between the means.")
