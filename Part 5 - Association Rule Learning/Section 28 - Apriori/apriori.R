# """ Which food purchases come together? """


# Create sparse atrix
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header=FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=10)

# Trainin Apriori on the dataset
rules = apriori(data=dataset,
                parameter = list(support = 0.004, confidence = 0.2))

# Visualize the results
inspect(sort(rules, by = 'lift')[1:10])