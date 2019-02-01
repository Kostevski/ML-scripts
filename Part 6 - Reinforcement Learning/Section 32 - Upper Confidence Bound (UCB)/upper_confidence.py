# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math

# Variables
N = 10000 # Number of people exposed to ads
d = 10 # Number of ads
ads_selected = []

number_of_selections = [0] * d # How many times each ad was selected
sums_of_rewards = [0] * d # Sums  of reward for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        if (number_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualizing the results
print("Numbers of subjects: {}".format(N))
print("Numbers of ad clicks: {}".format(total_reward))
#plt.hist(ads_selected)
plt.bar([x for x in range (0,d)],
         number_of_selections,
         label = "Ad displays",
         color="blue",
         width=1,
         edgecolor='black')

plt.bar([x for x in range (0,d)],
         sums_of_rewards,
         label = "Ad clicks",
         color="red",
         width=1,
         edgecolor='black')

plt.title("Histogram with UDC algorithm")
plt.xlabel('Ads')
plt.ylabel("Frequency")
plt.legend()
plt.show()
# Random selection algorithm
"""

import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

total_reward=1242 """