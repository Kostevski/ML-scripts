# Upper Confidence Bound

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

#Implementing USC Algorithm
N = 10000
d = 10

ads_selected = integer()
number_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (number_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / number_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / number_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Implementing Random Selection #
#################################
# N = 10000
# d = 10
# ads_selected = integer(0)
# total_reward = 0
# for (n in 1:N) {
#   ad = sample(1:10, 1)
#   ads_selected = append(ads_selected, ad)
#   reward = dataset[n, ad]
#   total_reward = total_reward + reward
# }
# 
################################
#       Total reward ~ 1200   ##
################################
# Visualising the results

hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')
