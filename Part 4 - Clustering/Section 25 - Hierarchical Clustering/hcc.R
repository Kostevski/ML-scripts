# """ What do spenders at a mall have in common? """

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using dendogram to find the optimal amount of clusters
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fittin hierarchical clustering to the mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

library(cluster)
clusplot(X,
         y_hc,
         lines=0,
         shade=TRUE,
         color=TRUE,
         labels=2,
         plotchar=FALSE,
         span=TRUE,
         main = paste("Cluster of clients"),
         xlab = "Annual Income",
         ylab = "Spending score")