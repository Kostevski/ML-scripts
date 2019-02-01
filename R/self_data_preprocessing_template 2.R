# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set) 

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ Level,
             data = dataset)

# Fitting Polyreg Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ .,
             data = dataset)

#Visiualize lin-reg
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color="red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset), 
            color="blue")) +
  ggtitle("Truth or Bluff?") +
  xlab('Level') +
  ylab('Salary')

# Visual polyreg
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color="red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset), 
                color="blue")) +
  ggtitle("Truth or Bluff? (Polynoimial)") +
  xlab('Level') +
  ylab('Salary')

# Predicting new result with linreg
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Predicting new result with poly
ypoly__pred = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                               Level2 = 6.5^2,
                                               Level3 = 6.5^3,
                                               Level4 = 6.5^4))
