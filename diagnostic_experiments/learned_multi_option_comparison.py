import os
import numpy as np
import pandas as pd
import pickle as pkl 
from utils import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

def per_run(a):
    return np.mean(a, axis=0)
def confidence_interval(x, CI=0.95):
    return CI * np.std(x)/np.mean(x)

def smoothen(scalars, weight=0.8):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

multi_options_folder_list = ["multi_option_noptions2_nruns10_nepisodes100_nsteps1000_fixed_eta_0.3",
                        "multi_option_noptions2_nruns10_nepisodes100_nsteps1000_LINEAR_eta_distance_cityblock",
                        "multi_option_noptions2_nruns10_nepisodes100_nsteps1000_LINEAR_eta_distance_euclidean"]

with open(os.path.join(multi_options_folder_list[0], "history_steps.pkl"), "rb") as f:
    fixed_eta_history_steps = pkl.load(f)
with open(os.path.join(multi_options_folder_list[1], "history_steps.pkl"), "rb") as f:
    linear_eta_cityblock_history_steps = pkl.load(f)
with open(os.path.join(multi_options_folder_list[2], "history_steps.pkl"), "rb") as f:
    linear_eta_euclidean_history_steps = pkl.load(f)

f_eta = smoothen(per_run(fixed_eta_history_steps))
l_cityblock_eta = smoothen(per_run(linear_eta_cityblock_history_steps))
l_euclidean_eta = smoothen(per_run(linear_eta_euclidean_history_steps))
print("Fiexed eta:", f_eta)
# print("Smoothed Fiexed eta:", exponential_smoothing(f_eta))
print("Cityblock eta:", l_cityblock_eta)
# print("Smooth Cityblock eta:", exponential_smoothing(l_cityblock_eta))
print("Euclidean eta:", l_euclidean_eta)
# print("Smooth Euclidean eta:", exponential_smoothing(l_euclidean_eta))

CI = 0.95
nepisodes = 100
plt.plot(range(100), f_eta, color='blue', label='Fixed eta at 0.3')
ci = confidence_interval(f_eta, CI) #confidence interval
plt.fill_between(range(100), (f_eta-ci), (f_eta+ci), color='blue', alpha=0.5)

plt.plot(range(100), l_cityblock_eta, color='green', label='Linear Cityblock eta')
ci = confidence_interval(l_cityblock_eta, CI) #confidence interval
plt.fill_between(range(100), (l_cityblock_eta-ci), (l_cityblock_eta+ci), color='green', alpha=0.5)

plt.plot(range(100), l_euclidean_eta, color='red', label='Linear Euclidean eta')
ci = confidence_interval(l_euclidean_eta, CI) #confidence interval
plt.fill_between(range(100), (l_euclidean_eta-ci), (l_euclidean_eta+ci), color='red', alpha=0.5)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Avg steps")
plt.title("Learned Options")
plt.savefig("Graph1_Learned_multioption_comparison.jpg")
plt.close("all")


