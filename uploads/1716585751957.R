# Task       :Classification Abalone Dataset
# Names      : Steven Gitonga Nyaga
# Student_ID : 336555

# *********************************************** OBJECTIVE **************************************************************
# The primary intent of this experiment is to devise a classification system for Abalone ages and construct a 
# decision tree accordingly
# The abalone dataset comprises of 8 class attributes, with our focus being predominantly on "Rings",0 as this is what
# determines the age of the abalones. Hence we will examine and decern whilch abalone is young, an adult, mature or old
# based on their age. This classification can be important in fisheries management, conservation efforts, and commercial 
# optimization. It aids in setting appropriate harvest quotas, protecting juveniles, and monitoring population health. 
# For commercial interests, age classification helps in pricing and breeding programs, 
# while for scientific research, it provides insights into growth rates and ecosystem health. 
# Accurate age prediction through models like Rpart allows for informed decision-making, ensuring the sustainability 
# and economic viability of abalone populations.

# clear work space and console
rm(list=ls())
cat("\014")

# Adding important libraries
library(party)
library(rpart)       # Decision tree training
library(rpart.plot)  # tree plotting
library(caret)       # Confusion Matrix
library(gmodels)     # Accuracy Metric
library(dplyr)       # For data manipulation

# Downloading and preparing dataset by adding columns and also checking for missing values
# download.file('http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', 'abalone.data')
abalone_ds <- read.table("abalone.data", header = FALSE, sep = ",", na.strings = "")
colnames(abalone_ds) <- c("Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings")
View(abalone_ds)
set.seed(1234)
str(abalone_ds)
# 'data.frame':	4177 obs. of  9 variables:
#   $ Sex           : chr  "M" "M" "F" "M" ...
# $ Length        : num  0.455 0.35 0.53 0.44 0.33 0.425 0.53 0.545 0.475 0.55 ...
# $ Diameter      : num  0.365 0.265 0.42 0.365 0.255 0.3 0.415 0.425 0.37 0.44 ...
# $ Height        : num  0.095 0.09 0.135 0.125 0.08 0.095 0.15 0.125 0.125 0.15 ...
# $ Whole_weight  : num  0.514 0.226 0.677 0.516 0.205 ...
# $ Shucked_weight: num  0.2245 0.0995 0.2565 0.2155 0.0895 ...
# $ Viscera_weight: num  0.101 0.0485 0.1415 0.114 0.0395 ...
# $ Shell_weight  : num  0.15 0.07 0.21 0.155 0.055 0.12 0.33 0.26 0.165 0.32 ...
# $ Rings         : int  15 7 9 10 7 8 20 16 9 19 ...
summary(abalone_ds)
# Sex                Length         Diameter          Height        Whole_weight    Shucked_weight   Viscera_weight  
# Length:4177        Min.   :0.075   Min.   :0.0550   Min.   :0.0000   Min.   :0.0020   Min.   :0.0010   Min.   :0.0005  
# Class :character   1st Qu.:0.450   1st Qu.:0.3500   1st Qu.:0.1150   1st Qu.:0.4415   1st Qu.:0.1860   1st Qu.:0.0935  
# Mode  :character   Median :0.545   Median :0.4250   Median :0.1400   Median :0.7995   Median :0.3360   Median :0.1710  
# Mean   :0.524   Mean   :0.4079   Mean   :0.1395   Mean   :0.8287   Mean   :0.3594   Mean   :0.1806  
# 3rd Qu.:0.615   3rd Qu.:0.4800   3rd Qu.:0.1650   3rd Qu.:1.1530   3rd Qu.:0.5020   3rd Qu.:0.2530  
# Max.   :0.815   Max.   :0.6500   Max.   :1.1300   Max.   :2.8255   Max.   :1.4880   Max.   :0.7600  
# Shell_weight        Rings       
# Min.   :0.0015   Min.   : 1.000  
# 1st Qu.:0.1300   1st Qu.: 8.000  
# Median :0.2340   Median : 9.000  
# Mean   :0.2388   Mean   : 9.934  
# 3rd Qu.:0.3290   3rd Qu.:11.000  
# Max.   :1.0050   Max.   :29.000
sum(is.na(abalone_ds))
# [1] 0 -> NO MISSING values

# checking distribution of the Rings column using histogram
hist(abalone_ds$Rings)
summary(abalone_ds$Rings)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.000   8.000   9.000   9.934  11.000  29.000 

# The aim here is to classify Abalones as "Young", "Adult", "Mature" and "Old".
# After testing different ranges I decided to settle with the one below to allow more less with the same number of records
# This is because as seen from the histogram above most of the abalones were between 5 and 15. Ensuring the number of 
# records are not too small is imprtant otherwise rpart can decide to not use that category in classification   
abalone_ds$Age <- cut(abalone_ds$Rings, breaks = c(0, 6, 7, 12, 30), labels = c("Young", "Adult", "Mature", "Old"))
View(abalone_ds)

# Randomly splitting the dataset into training set and testing set (70% and 30% of samples respectively)
s <- sample(2, nrow(abalone_ds), replace=TRUE, prob=c(0.7, 0.3))
trainData <- abalone_ds[s == 1,]
testData <- abalone_ds[s == 2,]
table(trainData$Age)
# Young  Adult Mature    Old 
# 329    275   1842    501 
table(testData$Age)
# Young  Adult Mature    Old 
# 119    116    803    192

# *********************************************** CLASSIFICATION **********************************************************

# Age is the class variable and we take remaining 8 features as independent variables
myFormula <- Age ~ Sex + Length + Diameter + Height + Whole_weight + Shucked_weight + Viscera_weight + Shell_weight

# Using Rpart
abalone_model <- rpart(myFormula,  method = "class", data = trainData)
# printcp(abalone_model)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class")
# 
# Variables actually used in tree construction:
#   [1] Sex            Shell_weight   Shucked_weight
# 
# Root node error: 1105/2947 = 0.37496
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.107692      0   1.00000 1.00000 0.023783
# 2 0.050679      1   0.89231 0.89231 0.023181
# 3 0.029864      2   0.84163 0.87511 0.023067
# 4 0.025792      3   0.81176 0.86787 0.023018
# 5 0.010000      5   0.76018 0.82081 0.022676
plotcp(abalone_model)
summary(abalone_model)
# Call:
#   rpart(formula = myFormula, data = trainData, method = "class")
# n= 2947 
# 
# CP nsplit rel error    xerror       xstd
# 1 0.10769231      0 1.0000000 1.0000000 0.02378338
# 2 0.05067873      1 0.8923077 0.8923077 0.02318061
# 3 0.02986425      2 0.8416290 0.8751131 0.02306713
# 4 0.02579186      3 0.8117647 0.8678733 0.02301787
# 5 0.01000000      5 0.7601810 0.8208145 0.02267600
# 
# Variable importance
# Shell_weight   Whole_weight       Diameter         Length Viscera_weight Shucked_weight            Sex         Height 
# 19             17             16             16             15             15              1              1 
# 
# Node number 1: 2947 observations,    complexity param=0.1076923
# predicted class=Mature  expected loss=0.3749576  P(node) =1
# class counts:   329   275  1842   501
# probabilities: 0.112 0.093 0.625 0.170 
# left son=2 (682 obs) right son=3 (2265 obs)
# Primary splits:
#   Shell_weight < 0.11975 to the left,  improve=261.1471, (0 missing)
# Height       < 0.1025  to the left,  improve=248.8652, (0 missing)
# Whole_weight < 0.34375 to the left,  improve=243.2824, (0 missing)
# Diameter     < 0.3375  to the left,  improve=240.0084, (0 missing)
# Length       < 0.4075  to the left,  improve=238.9212, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.39425 to the left,  agree=0.970, adj=0.870, (0 split)
# Diameter       < 0.3375  to the left,  agree=0.968, adj=0.862, (0 split)
# Length         < 0.4375  to the left,  agree=0.962, adj=0.837, (0 split)
# Viscera_weight < 0.08025 to the left,  agree=0.956, adj=0.808, (0 split)
# Shucked_weight < 0.16475 to the left,  agree=0.945, adj=0.762, (0 split)
# 
# Node number 2: 682 observations,    complexity param=0.05067873
# predicted class=Young   expected loss=0.5513196  P(node) =0.2314218
# class counts:   306   184   187     5
# probabilities: 0.449 0.270 0.274 0.007 
# left son=4 (242 obs) right son=5 (440 obs)
# Primary splits:
#   Shell_weight   < 0.0555  to the left,  improve=66.87098, (0 missing)
# Diameter       < 0.2225  to the left,  improve=61.75984, (0 missing)
# Whole_weight   < 0.16975 to the left,  improve=60.17031, (0 missing)
# Viscera_weight < 0.03375 to the left,  improve=58.74652, (0 missing)
# Height         < 0.0825  to the left,  improve=58.29924, (0 missing)
# Surrogate splits:
#   Diameter       < 0.2525  to the left,  agree=0.947, adj=0.851, (0 split)
# Whole_weight   < 0.1805  to the left,  agree=0.944, adj=0.843, (0 split)
# Length         < 0.3425  to the left,  agree=0.931, adj=0.806, (0 split)
# Shucked_weight < 0.06575 to the left,  agree=0.906, adj=0.736, (0 split)
# Viscera_weight < 0.03425 to the left,  agree=0.902, adj=0.723, (0 split)
# 
# Node number 3: 2265 observations,    complexity param=0.02579186
# predicted class=Mature  expected loss=0.2693157  P(node) =0.7685782
# class counts:    23    91  1655   496
# probabilities: 0.010 0.040 0.731 0.219 
# left son=6 (1850 obs) right son=7 (415 obs)
# Primary splits:
#   Shell_weight   < 0.3895  to the left,  improve=35.308130, (0 missing)
# Height         < 0.1725  to the left,  improve=20.696400, (0 missing)
# Whole_weight   < 0.71    to the left,  improve=12.351700, (0 missing)
# Viscera_weight < 0.15575 to the left,  improve=10.584910, (0 missing)
# Diameter       < 0.4025  to the left,  improve= 9.418724, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.427   to the left,  agree=0.927, adj=0.600, (0 split)
# Diameter       < 0.5125  to the left,  agree=0.914, adj=0.530, (0 split)
# Length         < 0.6475  to the left,  agree=0.906, adj=0.487, (0 split)
# Viscera_weight < 0.31775 to the left,  agree=0.888, adj=0.390, (0 split)
# Height         < 0.1875  to the left,  agree=0.882, adj=0.354, (0 split)
# 
# Node number 4: 242 observations
# predicted class=Young   expected loss=0.2066116  P(node) =0.08211741
# class counts:   192    33    17     0
# probabilities: 0.793 0.136 0.070 0.000 
# 
# Node number 5: 440 observations,    complexity param=0.02986425
# predicted class=Mature  expected loss=0.6136364  P(node) =0.1493044
# class counts:   114   151   170     5
# probabilities: 0.259 0.343 0.386 0.011 
# left son=10 (322 obs) right son=11 (118 obs)
# Primary splits:
#   Sex            splits as  RLR,         improve=18.281110, (0 missing)
# Height         < 0.1025  to the left,  improve= 7.068381, (0 missing)
# Shell_weight   < 0.09175 to the left,  improve= 5.944407, (0 missing)
# Diameter       < 0.2775  to the left,  improve= 3.001521, (0 missing)
# Viscera_weight < 0.03375 to the left,  improve= 2.397569, (0 missing)
# Surrogate splits:
#   Length         < 0.4725  to the left,  agree=0.734, adj=0.008, (0 split)
# Shucked_weight < 0.062   to the right, agree=0.734, adj=0.008, (0 split)
# 
# Node number 6: 1850 observations
# predicted class=Mature  expected loss=0.2340541  P(node) =0.627757
# class counts:    23    91  1417   319
# probabilities: 0.012 0.049 0.766 0.172 
# 
# Node number 7: 415 observations,    complexity param=0.02579186
# predicted class=Mature  expected loss=0.426506  P(node) =0.1408212
# class counts:     0     0   238   177
# probabilities: 0.000 0.000 0.573 0.427 
# left son=14 (302 obs) right son=15 (113 obs)
# Primary splits:
#   Shucked_weight < 0.56675 to the right, improve=32.945950, (0 missing)
# Viscera_weight < 0.28625 to the right, improve=14.352770, (0 missing)
# Whole_weight   < 1.34775 to the right, improve=11.706260, (0 missing)
# Length         < 0.6375  to the right, improve=10.420480, (0 missing)
# Diameter       < 0.4875  to the right, improve= 7.054752, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.369   to the right, agree=0.880, adj=0.558, (0 split)
# Length         < 0.6325  to the right, agree=0.841, adj=0.416, (0 split)
# Viscera_weight < 0.29525 to the right, agree=0.812, adj=0.310, (0 split)
# Diameter       < 0.5025  to the right, agree=0.788, adj=0.221, (0 split)
# Height         < 0.1525  to the right, agree=0.733, adj=0.018, (0 split)
# 
# Node number 10: 322 observations
# predicted class=Adult   expected loss=0.6086957  P(node) =0.1092637
# class counts:   102   126    93     1
# probabilities: 0.317 0.391 0.289 0.003 
# 
# Node number 11: 118 observations
# predicted class=Mature  expected loss=0.3474576  P(node) =0.04004072
# class counts:    12    25    77     4
# probabilities: 0.102 0.212 0.653 0.034 
# 
# Node number 14: 302 observations
# predicted class=Mature  expected loss=0.3046358  P(node) =0.1024771
# class counts:     0     0   210    92
# probabilities: 0.000 0.000 0.695 0.305 
# 
# Node number 15: 113 observations
# predicted class=Old     expected loss=0.2477876  P(node) =0.03834408
# class counts:     0     0    28    85
# probabilities: 0.000 0.000 0.248 0.752 
rpart.plot(abalone_model, main="Classification for Abalone")

# classification of the training data
abalone_trainPred = predict(abalone_model,trainData,type = "class")
table(abalone_trainPred, trainData$Age)
# confusionMatrix(abalone_trainPred, trainData$Age, mode="everything")
train_conf_matrix <- confusionMatrix(abalone_trainPred, trainData$Age, mode = "everything")
train_conf_matrix
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Young Adult Mature  Old
# Young    192    33     17    0
# Adult    102   126     93    1
# Mature    35   116   1704  415
# Old        0     0     28   85
# 
# Overall Statistics
# 
# Accuracy : 0.715           
# 95% CI : (0.6983, 0.7312)
# No Information Rate : 0.625           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4214          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Young Class: Adult Class: Mature Class: Old
# Sensitivity               0.58359      0.45818        0.9251    0.16966
# Specificity               0.98090      0.92665        0.4878    0.98855
# Pos Pred Value            0.79339      0.39130        0.7507    0.75221
# Neg Pred Value            0.94935      0.94324        0.7962    0.85321
# Precision                 0.79339      0.39130        0.7507    0.75221
# Recall                    0.58359      0.45818        0.9251    0.16966
# F1                        0.67250      0.42211        0.8288    0.27687
# Prevalence                0.11164      0.09332        0.6250    0.17000
# Detection Rate            0.06515      0.04276        0.5782    0.02884
# Detection Prevalence      0.08212      0.10926        0.7703    0.03834
# Balanced Accuracy         0.78224      0.69241        0.7064    0.57911
mean(abalone_trainPred == trainData$Age)
# [1] 0.7149644
error <- mean(abalone_trainPred != trainData$Age)
error
# [1] 0.2850356

# Extract recall and compute average recall
train_recall <- train_conf_matrix$byClass[, "Recall"]
average_train_recall <- mean(train_recall, na.rm = TRUE)
# Compute overall accuracy
train_accuracy <- train_conf_matrix$overall["Accuracy"]
average_train_recall
# [1] 0.5341276
train_accuracy
# Accuracy 
# 0.7149644

# The accuracy is approximately 71.5% and the training average recall is approximately 53.4% meaning hat the 
# model is correctly identifying about 53.4% of the instances across all classes. 


# classification of the testing data
abalone_testPred = predict(abalone_model,testData,type = "class")
table(abalone_testPred, testData$Age)
train_conf_matrix <- confusionMatrix(abalone_testPred, testData$Age, mode = "everything")
train_conf_matrix
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Young Adult Mature Old
# Young     73    18     12   1
# Adult     35    39     29   2
# Mature    11    59    744 160
# Old        0     0     18  29
# 
# Overall Statistics
# 
# Accuracy : 0.7195          
# 95% CI : (0.6935, 0.7445)
# No Information Rate : 0.6528          
# P-Value [Acc > NIR] : 3.485e-07       
# 
# Kappa : 0.3913          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Young Class: Adult Class: Mature Class: Old
# Sensitivity               0.61345      0.33621        0.9265    0.15104
# Specificity               0.97210      0.94075        0.4614    0.98266
# Pos Pred Value            0.70192      0.37143        0.7639    0.61702
# Neg Pred Value            0.95915      0.93156        0.7695    0.86221
# Precision                 0.70192      0.37143        0.7639    0.61702
# Recall                    0.61345      0.33621        0.9265    0.15104
# F1                        0.65471      0.35294        0.8374    0.24268
# Prevalence                0.09675      0.09431        0.6528    0.15610
# Detection Rate            0.05935      0.03171        0.6049    0.02358
# Detection Prevalence      0.08455      0.08537        0.7919    0.03821
# Balanced Accuracy         0.79277      0.63848        0.6939    0.56685
mean(abalone_testPred == testData$Age)
# [1] 0.7195122
error <- mean(abalone_testPred != testData$Age)
error
# [1] 0.2804878

# Extract recall and compute average recall
train_recall <- train_conf_matrix$byClass[, "Recall"]
average_train_recall <- mean(train_recall, na.rm = TRUE)
# Compute overall accuracy
train_accuracy <- train_conf_matrix$overall["Accuracy"]
average_train_recall
# [1] 0.5068049
train_accuracy
# Accuracy 
# 0.7195122

# The accuracy is approximately 71.95% and the training average recall is approximately 50.68%

# The above are satisfactory accurancies but can go deeper to explore and potentially find the optimal configuration

# Using Loss Matrix
lossM_refined <- matrix(c(0, 1, 2, 3,   # Predicted: Young (increased cost for misclassifying )
                          1, 0, 1, 2,   # Predicted: Adult (increased cost for misclassifying )
                          2, 1, 0, 1,   # Predicted: Mature (increased cost for misclassifying )
                          3, 2, 1, 0),  # Predicted: Old (increased cost for misclassifying )
                        byrow = TRUE, nrow = 4)
lossM_refined

abalone_modelLM <-  rpart(myFormula,  method="class", data=trainData, parms = list(loss = lossM_refined ))
printcp(abalone_modelLM)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(loss = lossM_refined))
# 
# Variables actually used in tree construction:
#   [1] Sex            Shell_weight   Shucked_weight
# 
# Root node error: 1434/2947 = 0.4866
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.207810      0   1.00000 1.00000 0.026039
# 2 0.099024      1   0.79219 0.80265 0.019767
# 3 0.030683      2   0.69317 0.71130 0.019513
# 4 0.019526      3   0.66248 0.69317 0.019560
# 5 0.010000      5   0.62343 0.66179 0.019418
plotcp(abalone_modelLM)
summary(abalone_modelLM)
rpart.plot(abalone_modelLM, main="Classification for Abalone")
abalone_trainPred = predict(abalone_modelLM,trainData,type = "class")
table(abalone_trainPred, trainData$Age)
train_conf_matrix <- confusionMatrix(abalone_trainPred, trainData$Age, mode = "everything")
train_conf_matrix
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Young Adult Mature  Old
# Young    192    33     17    0
# Adult    102   126     93    1
# Mature    35   116   1687  399
# Old        0     0     45  101
# 
# Overall Statistics
# 
# Accuracy : 0.7146          
# 95% CI : (0.6979, 0.7309)
# No Information Rate : 0.625           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4267          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Young Class: Adult Class: Mature Class: Old
# Sensitivity               0.58359      0.45818        0.9159    0.20160
# Specificity               0.98090      0.92665        0.5023    0.98160
# Pos Pred Value            0.79339      0.39130        0.7541    0.69178
# Neg Pred Value            0.94935      0.94324        0.7817    0.85719
# Precision                 0.79339      0.39130        0.7541    0.69178
# Recall                    0.58359      0.45818        0.9159    0.20160
# F1                        0.67250      0.42211        0.8272    0.31221
# Prevalence                0.11164      0.09332        0.6250    0.17000
# Detection Rate            0.06515      0.04276        0.5724    0.03427
# Detection Prevalence      0.08212      0.10926        0.7591    0.04954
# Balanced Accuracy         0.78224      0.69241        0.7091    0.59160
mean(abalone_trainPred == trainData$Age)
# 0.6491347
error <- mean(abalone_trainPred != trainData$Age)
error
# 0.3508653

# Extract recall and compute average recall
train_recall <- train_conf_matrix$byClass[, "Recall"]
average_train_recall <- mean(train_recall, na.rm = TRUE)
# Compute overall accuracy
train_accuracy <- train_conf_matrix$overall["Accuracy"]
average_train_recall
# [1] 0.5398044
train_accuracy
# Accuracy 
# 0.714625

# The accuracy is approximately 71.46% and the training average recall is approximately 53.98%


# Trying different values for our loss matrix
lossM_refined <- matrix(c(0, 1, 1, 2,   # Predicted: Young (increased cost for misclassifying )
                          1, 0, 2, 1,   # Predicted: Adult (increased cost for misclassifying )
                          1, 2, 0, 4,   # Predicted: Mature (increased cost for misclassifying )
                          2, 1, 4, 0),  # Predicted: Old (increased cost for misclassifying )
                        byrow = TRUE, nrow = 4)
lossM_refined

abalone_modelLM <-  rpart(myFormula,  method="class", data=trainData, parms = list(loss = lossM_refined ))
printcp(abalone_modelLM)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(loss = lossM_refined))
# 
# Variables actually used in tree construction:
#   [1] Diameter       Shell_weight   Shucked_weight
# 
# Root node error: 2883/2947 = 0.97828
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.104058      0   1.00000 1.00000 0.028232
# 2 0.040236      1   0.89594 0.90080 0.028210
# 3 0.038848      2   0.85571 0.89629 0.027545
# 4 0.023933      3   0.81686 0.85709 0.026902
# 5 0.015088      5   0.76899 0.82206 0.025357
# 6 0.014395      7   0.73881 0.80541 0.023450
# 7 0.010000      9   0.71002 0.79050 0.024182
plotcp(abalone_modelLM)
summary(abalone_modelLM)
rpart.plot(abalone_modelLM, main="Classification for Abalone")
abalone_trainPred = predict(abalone_modelLM,trainData,type = "class")
table(abalone_trainPred, trainData$Age)
train_conf_matrix <- confusionMatrix(abalone_trainPred, trainData$Age, mode = "everything")
train_conf_matrix
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Young Adult Mature  Old
# Young    278   121    103    2
# Adult      2     2    163  210
# Mature    49   152   1548  204
# Old        0     0     28   85
# 
# Overall Statistics
# 
# Accuracy : 0.6491          
# 95% CI : (0.6316, 0.6664)
# No Information Rate : 0.625           
# P-Value [Acc > NIR] : 0.00354         
# 
# Kappa : 0.36            
# 
# Mcnemar's Test P-Value : < 2e-16         
# 
# Statistics by Class:
# 
#                      Class: Young Class: Adult Class: Mature Class: Old
# Sensitivity               0.84498    0.0072727        0.8404    0.16966
# Specificity               0.91367    0.8596557        0.6335    0.98855
# Pos Pred Value            0.55159    0.0053050        0.7926    0.75221
# Neg Pred Value            0.97912    0.8937743        0.7042    0.85321
# Precision                 0.55159    0.0053050        0.7926    0.75221
# Recall                    0.84498    0.0072727        0.8404    0.16966
# F1                        0.66747    0.0061350        0.8158    0.27687
# Prevalence                0.11164    0.0933152        0.6250    0.17000
# Detection Rate            0.09433    0.0006787        0.5253    0.02884
# Detection Prevalence      0.17102    0.1279267        0.6627    0.03834
mean(abalone_trainPred == trainData$Age)
# 0.6491347
error <- mean(abalone_trainPred != trainData$Age)
error
# 0.3508653

# Extract recall and compute average recall
train_recall <- train_conf_matrix$byClass[, "Recall"]
average_train_recall <- mean(train_recall, na.rm = TRUE)
# Compute overall accuracy
train_accuracy <- train_conf_matrix$overall["Accuracy"]
average_train_recall
# [1] 0.4655773
train_accuracy
# Accuracy 
# 0.6491347

# The accuracy is approximately 64.91% and the training average recall is approximately 46.55% -> Worst result yet

# Trying different values for the Loss matrix
lossM_refined <- matrix(c(0, 1, 1, 2,   # Predicted: Young
                          1, 0, 2, 1,   # Predicted: Adult
                          2, 3, 0, 4,   # Predicted: Mature (increased cost for misclassifying as Old)
                          3, 2, 4, 0),  # Predicted: Old (increased cost for misclassifying as Mature)
                        byrow = TRUE, nrow = 4)
lossM_refined

abalone_modelLM <-  rpart(myFormula,  method="class", data=trainData, parms = list(loss = lossM_refined ))
printcp(abalone_modelLM)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(loss = lossM_refined))
# 
# Variables actually used in tree construction:
#   [1] Shell_weight   Shucked_weight
# 
# Root node error: 2883/2947 = 0.97828
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.074228      0   1.00000 1.20950 0.030399
# 2 0.038848      1   0.92577 1.03087 0.029932
# 3 0.020118      3   0.84807 0.97260 0.029235
# 4 0.010000      5   0.80784 0.96566 0.029149
plotcp(abalone_modelLM)
summary(abalone_modelLM)
rpart.plot(abalone_modelLM, main="Classification for Abalone")
abalone_trainPred = predict(abalone_modelLM,trainData,type = "class")
table(abalone_trainPred, trainData$Age)
train_conf_matrix <- confusionMatrix(abalone_trainPred, trainData$Age, mode = "everything")
train_conf_matrix
mean(abalone_trainPred == trainData$Age)
# [1] 0.7170003
error <- mean(abalone_trainPred != trainData$Age)
error
# [1] 0.2829997

# Extract recall and compute average recall
train_recall <- train_conf_matrix$byClass[, "Recall"]
average_train_recall <- mean(train_recall, na.rm = TRUE)
# Compute overall accuracy
train_accuracy <- train_conf_matrix$overall["Accuracy"]
average_train_recall
# [1] 0.4782238
train_accuracy
# Accuracy 
# 0.7170003

# The accuracy is approximately 71.7% and the training average recall is approximately 47.82%

# Results observations:
# Without Loss matrix:
# Accuracy: 71.5% & Recall 53.4%
# With Loss matrix:
# Accuracy: 71.46% & Recall 53.98%
# Accuracy: 64.91% & Recall 46.55%
# Accuracy: 71.7% & Recall 47.82%

# The only other accuracy that is larger than 71.5% (without loss matrix) is  & Recall 47.82%
# but the Recall is quite low at 47.82%. Hence I will stick to the solution without loss matrix.

# Reverting to the initial solution without loss matrix
abalone_model <- rpart(myFormula,  method = "class", data = trainData)
abalone_trainPred = predict(abalone_model,trainData,type = "class")
abalone_testPred = predict(abalone_model,testData,type = "class")

# changing paramaters of rpart to minbucket = 60 and maxDepth to 2 to check their influence
rpControl = rpart.control(minbucket = 60, maxDepth = 2)
# training tree pruning
abalone_model_new <- rpart(myFormula,  method = "class", data = trainData, control = rpControl, parms = list(split = "information"))
pRp_abalone_model_new<- prune(abalone_model_new, cp = abalone_model_new$cptable[which.min(abalone_model_new$cptable[,"xerror"]),"CP"])
printcp(abalone_model_new)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# 
# Variables actually used in tree construction:
#   [1] Sex            Shell_weight   Shucked_weight
# 
# Root node error: 1105/2947 = 0.37496
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.107692      0   1.00000 1.00000 0.023783
# 2 0.050679      1   0.89231 0.88054 0.023103
# 3 0.029864      2   0.84163 0.86154 0.022974
# 4 0.016893      3   0.81176 0.83348 0.022772
# 5 0.010000      6   0.76109 0.81086 0.022599
summary(abalone_model_new)
# Call:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# n= 2947 
# 
# CP nsplit rel error    xerror       xstd
# 1 0.10769231      0 1.0000000 1.0000000 0.02378338
# 2 0.05067873      1 0.8923077 0.8805430 0.02310349
# 3 0.02986425      2 0.8416290 0.8615385 0.02297405
# 4 0.01689291      3 0.8117647 0.8334842 0.02277179
# 5 0.01000000      6 0.7610860 0.8108597 0.02259875
# 
# Variable importance
# Shell_weight   Whole_weight       Diameter         Length Viscera_weight Shucked_weight            Sex 
# 20             17             16             16             15             14              1 
# 
# Node number 1: 2947 observations,    complexity param=0.1076923
# predicted class=Mature  expected loss=0.3749576  P(node) =1
# class counts:   329   275  1842   501
# probabilities: 0.112 0.093 0.625 0.170 
# left son=2 (682 obs) right son=3 (2265 obs)
# Primary splits:
#   Shell_weight   < 0.11975 to the left,  improve=703.4058, (0 missing)
# Diameter       < 0.3425  to the left,  improve=642.2094, (0 missing)
# Whole_weight   < 0.42325 to the left,  improve=641.5147, (0 missing)
# Length         < 0.4525  to the left,  improve=619.1732, (0 missing)
# Viscera_weight < 0.11025 to the left,  improve=608.0687, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.39425 to the left,  agree=0.970, adj=0.870, (0 split)
# Diameter       < 0.3375  to the left,  agree=0.968, adj=0.862, (0 split)
# Length         < 0.4375  to the left,  agree=0.962, adj=0.837, (0 split)
# Viscera_weight < 0.08025 to the left,  agree=0.956, adj=0.808, (0 split)
# Shucked_weight < 0.16475 to the left,  agree=0.945, adj=0.762, (0 split)
# 
# Node number 2: 682 observations,    complexity param=0.05067873
# predicted class=Young   expected loss=0.5513196  P(node) =0.2314218
# class counts:   306   184   187     5
# probabilities: 0.449 0.270 0.274 0.007 
# left son=4 (242 obs) right son=5 (440 obs)
# Primary splits:
#   Shell_weight   < 0.0555  to the left,  improve=97.99399, (0 missing)
# Diameter       < 0.2225  to the left,  improve=93.53946, (0 missing)
# Whole_weight   < 0.16975 to the left,  improve=86.78150, (0 missing)
# Viscera_weight < 0.03375 to the left,  improve=85.26963, (0 missing)
# Length         < 0.2875  to the left,  improve=84.73763, (0 missing)
# Surrogate splits:
#   Diameter       < 0.2525  to the left,  agree=0.947, adj=0.851, (0 split)
# Whole_weight   < 0.1805  to the left,  agree=0.944, adj=0.843, (0 split)
# Length         < 0.3425  to the left,  agree=0.931, adj=0.806, (0 split)
# Shucked_weight < 0.06575 to the left,  agree=0.906, adj=0.736, (0 split)
# Viscera_weight < 0.03425 to the left,  agree=0.902, adj=0.723, (0 split)
# 
# Node number 3: 2265 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2693157  P(node) =0.7685782
# class counts:    23    91  1655   496
# probabilities: 0.010 0.040 0.731 0.219 
# left son=6 (492 obs) right son=7 (1773 obs)
# Primary splits:
#   Shell_weight   < 0.18975 to the left,  improve=122.30550, (0 missing)
# Height         < 0.1525  to the left,  improve= 76.71986, (0 missing)
# Whole_weight   < 0.80475 to the left,  improve= 73.19802, (0 missing)
# Diameter       < 0.4125  to the left,  improve= 72.27077, (0 missing)
# Viscera_weight < 0.16325 to the left,  improve= 68.90266, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.65975 to the left,  agree=0.932, adj=0.687, (0 split)
# Diameter       < 0.3975  to the left,  agree=0.919, adj=0.626, (0 split)
# Viscera_weight < 0.13625 to the left,  agree=0.906, adj=0.565, (0 split)
# Length         < 0.5175  to the left,  agree=0.900, adj=0.539, (0 split)
# Shucked_weight < 0.25575 to the left,  agree=0.880, adj=0.449, (0 split)
# 
# Node number 4: 242 observations
# predicted class=Young   expected loss=0.2066116  P(node) =0.08211741
# class counts:   192    33    17     0
# probabilities: 0.793 0.136 0.070 0.000 
# 
# Node number 5: 440 observations,    complexity param=0.02986425
# predicted class=Mature  expected loss=0.6136364  P(node) =0.1493044
# class counts:   114   151   170     5
# probabilities: 0.259 0.343 0.386 0.011 
# left son=10 (322 obs) right son=11 (118 obs)
# Primary splits:
#   Sex            splits as  RLR,         improve=30.125160, (0 missing)
# Height         < 0.1025  to the left,  improve=11.624350, (0 missing)
# Shell_weight   < 0.09175 to the left,  improve=10.753220, (0 missing)
# Diameter       < 0.2775  to the left,  improve= 4.834994, (0 missing)
# Viscera_weight < 0.04225 to the left,  improve= 4.390376, (0 missing)
# Surrogate splits:
#   Length         < 0.4725  to the left,  agree=0.734, adj=0.008, (0 split)
# Shucked_weight < 0.062   to the right, agree=0.734, adj=0.008, (0 split)
# 
# Node number 6: 492 observations
# predicted class=Mature  expected loss=0.245935  P(node) =0.1669494
# class counts:    17    71   371    33
# probabilities: 0.035 0.144 0.754 0.067 
# 
# Node number 7: 1773 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2758037  P(node) =0.6016288
# class counts:     6    20  1284   463
# probabilities: 0.003 0.011 0.724 0.261 
# left son=14 (1229 obs) right son=15 (544 obs)
# Primary splits:
#   Shell_weight   < 0.35975 to the left,  improve=42.67379, (0 missing)
# Height         < 0.1725  to the left,  improve=23.04216, (0 missing)
# Shucked_weight < 0.44475 to the right, improve=21.73438, (0 missing)
# Diameter       < 0.4875  to the left,  improve=13.18220, (0 missing)
# Length         < 0.5375  to the right, improve=12.47624, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.344   to the left,  agree=0.895, adj=0.656, (0 split)
# Diameter       < 0.4975  to the left,  agree=0.873, adj=0.585, (0 split)
# Length         < 0.6325  to the left,  agree=0.856, adj=0.531, (0 split)
# Height         < 0.1725  to the left,  agree=0.841, adj=0.482, (0 split)
# Viscera_weight < 0.30175 to the left,  agree=0.836, adj=0.465, (0 split)
# 
# Node number 10: 322 observations
# predicted class=Adult   expected loss=0.6086957  P(node) =0.1092637
# class counts:   102   126    93     1
# probabilities: 0.317 0.391 0.289 0.003 
# 
# Node number 11: 118 observations
# predicted class=Mature  expected loss=0.3474576  P(node) =0.04004072
# class counts:    12    25    77     4
# probabilities: 0.102 0.212 0.653 0.034 
# 
# Node number 14: 1229 observations
# predicted class=Mature  expected loss=0.2229455  P(node) =0.4170343
# class counts:     6    20   955   248
# probabilities: 0.005 0.016 0.777 0.202 
# 
# Node number 15: 544 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.3952206  P(node) =0.1845945
# class counts:     0     0   329   215
# probabilities: 0.000 0.000 0.605 0.395 
# left son=30 (398 obs) right son=31 (146 obs)
# Primary splits:
#   Shucked_weight < 0.53525 to the right, improve=36.48604, (0 missing)
# Viscera_weight < 0.28725 to the right, improve=20.99502, (0 missing)
# Whole_weight   < 1.19525 to the right, improve=15.90028, (0 missing)
# Length         < 0.6225  to the right, improve=11.40289, (0 missing)
# Diameter       < 0.4875  to the right, improve=10.14492, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.32875 to the right, agree=0.888, adj=0.582, (0 split)
# Length         < 0.6325  to the right, agree=0.846, adj=0.425, (0 split)
# Viscera_weight < 0.26375 to the right, agree=0.840, adj=0.404, (0 split)
# Diameter       < 0.4925  to the right, agree=0.807, adj=0.281, (0 split)
# Height         < 0.1575  to the right, agree=0.752, adj=0.075, (0 split)
# 
# Node number 30: 398 observations
# predicted class=Mature  expected loss=0.2864322  P(node) =0.1350526
# class counts:     0     0   284   114
# probabilities: 0.000 0.000 0.714 0.286 
# 
# Node number 31: 146 observations
# predicted class=Old     expected loss=0.3082192  P(node) =0.04954191
# class counts:     0     0    45   101
# probabilities: 0.000 0.000 0.308 0.692 
plotcp(abalone_model_new)
rpart.plot(pRp_abalone_model_new, main="Classification for Abalone")


# Test pruning
abalone_model_new <- rpart(myFormula,  method = "class", data = testData, control = rpControl, parms = list(split = "information"))
pRp_abalone_model_new<- prune(abalone_model_new, cp = abalone_model_new$cptable[which.min(abalone_model_new$cptable[,"xerror"]),"CP"])
printcp(pRp_abalone_model_new)
# Classification tree:
#   rpart(formula = myFormula, data = testData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# 
# Variables actually used in tree construction:
#   [1] Sex          Shell_weight
# 
# Root node error: 427/1230 = 0.34715
# 
# n= 1230 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.064403      0   1.00000 1.00000 0.039101
# 2 0.032787      2   0.87119 0.89461 0.038006
# 3 0.010000      3   0.83841 0.88056 0.037839
plotcp(pRp_abalone_model_new)
summary(pRp_abalone_model_new)
# Call:
#   rpart(formula = myFormula, data = testData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# n= 1230 
# 
# CP nsplit rel error    xerror       xstd
# 1 0.06440281      0 1.0000000 1.0000000 0.03910131
# 2 0.03278689      2 0.8711944 0.8946136 0.03800574
# 3 0.01000000      3 0.8384075 0.8805621 0.03783924
# 
# Variable importance
# Shell_weight   Whole_weight       Diameter         Length Viscera_weight         Height Shucked_weight            Sex 
# 20             17             16             16             15             13              2              2 
# 
# Node number 1: 1230 observations,    complexity param=0.06440281
# predicted class=Mature  expected loss=0.3471545  P(node) =1
# class counts:   119   116   803   192
# probabilities: 0.097 0.094 0.653 0.156 
# left son=2 (343 obs) right son=3 (887 obs)
# Primary splits:
#   Shell_weight   < 0.1445  to the left,  improve=265.6787, (0 missing)
# Height         < 0.1175  to the left,  improve=241.7600, (0 missing)
# Diameter       < 0.3525  to the left,  improve=230.0928, (0 missing)
# Length         < 0.4375  to the left,  improve=228.6349, (0 missing)
# Viscera_weight < 0.0995  to the left,  improve=225.5560, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.496   to the left,  agree=0.955, adj=0.840, (0 split)
# Diameter       < 0.3575  to the left,  agree=0.950, adj=0.819, (0 split)
# Length         < 0.4675  to the left,  agree=0.943, adj=0.796, (0 split)
# Viscera_weight < 0.09575 to the left,  agree=0.937, adj=0.773, (0 split)
# Height         < 0.1175  to the left,  agree=0.936, adj=0.770, (0 split)
# 
# Node number 2: 343 observations,    complexity param=0.06440281
# predicted class=Mature  expected loss=0.6326531  P(node) =0.2788618
# class counts:   114    94   126     9
# probabilities: 0.332 0.274 0.367 0.026 
# left son=4 (65 obs) right son=5 (278 obs)
# Primary splits:
#   Shell_weight   < 0.0395  to the left,  improve=54.92510, (0 missing)
# Viscera_weight < 0.03225 to the left,  improve=53.67642, (0 missing)
# Diameter       < 0.2225  to the left,  improve=49.62583, (0 missing)
# Whole_weight   < 0.1555  to the left,  improve=46.42627, (0 missing)
# Length         < 0.3075  to the left,  improve=45.78475, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.12475 to the left,  agree=0.971, adj=0.846, (0 split)
# Length         < 0.2925  to the left,  agree=0.965, adj=0.815, (0 split)
# Viscera_weight < 0.02825 to the left,  agree=0.965, adj=0.815, (0 split)
# Diameter       < 0.2225  to the left,  agree=0.962, adj=0.800, (0 split)
# Shucked_weight < 0.04775 to the left,  agree=0.948, adj=0.723, (0 split)
# 
# Node number 3: 887 observations
# predicted class=Mature  expected loss=0.2367531  P(node) =0.7211382
# class counts:     5    22   677   183
# probabilities: 0.006 0.025 0.763 0.206 
# 
# Node number 4: 65 observations
# predicted class=Young   expected loss=0.1230769  P(node) =0.05284553
# class counts:    57     6     2     0
# probabilities: 0.877 0.092 0.031 0.000 
# 
# Node number 5: 278 observations,    complexity param=0.03278689
# predicted class=Mature  expected loss=0.5539568  P(node) =0.2260163
# class counts:    57    88   124     9
# probabilities: 0.205 0.317 0.446 0.032 
# left son=10 (179 obs) right son=11 (99 obs)
# Primary splits:
#   Sex            splits as  RLR,         improve=24.590640, (0 missing)
# Shell_weight   < 0.11025 to the left,  improve=18.472970, (0 missing)
# Height         < 0.1125  to the left,  improve=10.077110, (0 missing)
# Diameter       < 0.2725  to the left,  improve=10.067690, (0 missing)
# Viscera_weight < 0.07275 to the left,  improve= 9.434992, (0 missing)
# Surrogate splits:
#   Viscera_weight < 0.095   to the left,  agree=0.676, adj=0.091, (0 split)
# Shucked_weight < 0.05275 to the right, agree=0.655, adj=0.030, (0 split)
# Diameter       < 0.3975  to the left,  agree=0.651, adj=0.020, (0 split)
# Height         < 0.0675  to the right, agree=0.651, adj=0.020, (0 split)
# Whole_weight   < 0.51625 to the left,  agree=0.651, adj=0.020, (0 split)
# 
# Node number 10: 179 observations
# predicted class=Adult   expected loss=0.6145251  P(node) =0.1455285
# class counts:    51    69    55     4
# probabilities: 0.285 0.385 0.307 0.022 
# 
# Node number 11: 99 observations
# predicted class=Mature  expected loss=0.3030303  P(node) =0.0804878
# class counts:     6    19    69     5
# probabilities: 0.061 0.192 0.697 0.051
rpart.plot(pRp_abalone_model_new, main="Classification for Abalone")


# changing minbucket to 30 
rpControl = rpart.control(minbucket = 30, maxDepth = 2);
pRp_abalone_model_new <- rpart(myFormula,  method = "class", data = trainData, control = rpControl, parms = list(split = "information"))
pRp_abalone_model_new<- prune(pRp_abalone_model_new, cp = pRp_abalone_model_new$cptable[which.min(pRp_abalone_model_new$cptable[,"xerror"]),"CP"])
printcp(pRp_abalone_model_new)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# 
# Variables actually used in tree construction:
#   [1] Sex            Shell_weight   Shucked_weight
# 
# Root node error: 1105/2947 = 0.37496
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.107692      0   1.00000 1.00000 0.023783
# 2 0.050679      1   0.89231 0.89231 0.023181
# 3 0.029864      2   0.84163 0.87330 0.023055
# 4 0.016893      3   0.81176 0.82986 0.022745
# 5 0.013122      6   0.76109 0.79457 0.022469
plotcp(pRp_abalone_model_new)
summary(pRp_abalone_model_new)
# Call:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# n= 2947 
# 
# CP nsplit rel error    xerror       xstd
# 1 0.10769231      0 1.0000000 1.0000000 0.02378338
# 2 0.05067873      1 0.8923077 0.8868778 0.02314530
# 3 0.02986425      2 0.8416290 0.8624434 0.02298035
# 4 0.01689291      3 0.8117647 0.8280543 0.02273108
# 5 0.01312217      6 0.7610860 0.7945701 0.02246856
# 6 0.01000000      8 0.7348416 0.7846154 0.02238664
# 
# Variable importance
# Shell_weight   Whole_weight       Diameter         Length Viscera_weight Shucked_weight            Sex         Height 
# 20             17             16             16             15             15              1              1 
# 
# Node number 1: 2947 observations,    complexity param=0.1076923
# predicted class=Mature  expected loss=0.3749576  P(node) =1
# class counts:   329   275  1842   501
# probabilities: 0.112 0.093 0.625 0.170 
# left son=2 (682 obs) right son=3 (2265 obs)
# Primary splits:
#   Shell_weight   < 0.11975 to the left,  improve=703.4058, (0 missing)
# Diameter       < 0.3425  to the left,  improve=642.2094, (0 missing)
# Whole_weight   < 0.42325 to the left,  improve=641.5147, (0 missing)
# Length         < 0.4525  to the left,  improve=619.1732, (0 missing)
# Viscera_weight < 0.11025 to the left,  improve=608.0687, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.39425 to the left,  agree=0.970, adj=0.870, (0 split)
# Diameter       < 0.3375  to the left,  agree=0.968, adj=0.862, (0 split)
# Length         < 0.4375  to the left,  agree=0.962, adj=0.837, (0 split)
# Viscera_weight < 0.08025 to the left,  agree=0.956, adj=0.808, (0 split)
# Shucked_weight < 0.16475 to the left,  agree=0.945, adj=0.762, (0 split)
# 
# Node number 2: 682 observations,    complexity param=0.05067873
# predicted class=Young   expected loss=0.5513196  P(node) =0.2314218
# class counts:   306   184   187     5
# probabilities: 0.449 0.270 0.274 0.007 
# left son=4 (242 obs) right son=5 (440 obs)
# Primary splits:
#   Shell_weight   < 0.0555  to the left,  improve=97.99399, (0 missing)
# Diameter       < 0.2225  to the left,  improve=93.53946, (0 missing)
# Whole_weight   < 0.16975 to the left,  improve=86.78150, (0 missing)
# Viscera_weight < 0.03375 to the left,  improve=85.26963, (0 missing)
# Length         < 0.2875  to the left,  improve=84.73763, (0 missing)
# Surrogate splits:
#   Diameter       < 0.2525  to the left,  agree=0.947, adj=0.851, (0 split)
# Whole_weight   < 0.1805  to the left,  agree=0.944, adj=0.843, (0 split)
# Length         < 0.3425  to the left,  agree=0.931, adj=0.806, (0 split)
# Shucked_weight < 0.06575 to the left,  agree=0.906, adj=0.736, (0 split)
# Viscera_weight < 0.03425 to the left,  agree=0.902, adj=0.723, (0 split)
# 
# Node number 3: 2265 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2693157  P(node) =0.7685782
# class counts:    23    91  1655   496
# probabilities: 0.010 0.040 0.731 0.219 
# left son=6 (492 obs) right son=7 (1773 obs)
# Primary splits:
#   Shell_weight   < 0.18975 to the left,  improve=122.30550, (0 missing)
# Height         < 0.1525  to the left,  improve= 76.71986, (0 missing)
# Whole_weight   < 0.80475 to the left,  improve= 73.19802, (0 missing)
# Diameter       < 0.4125  to the left,  improve= 72.27077, (0 missing)
# Viscera_weight < 0.16325 to the left,  improve= 68.90266, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.65975 to the left,  agree=0.932, adj=0.687, (0 split)
# Diameter       < 0.3975  to the left,  agree=0.919, adj=0.626, (0 split)
# Viscera_weight < 0.13625 to the left,  agree=0.906, adj=0.565, (0 split)
# Length         < 0.5175  to the left,  agree=0.900, adj=0.539, (0 split)
# Shucked_weight < 0.25575 to the left,  agree=0.880, adj=0.449, (0 split)
# 
# Node number 4: 242 observations
# predicted class=Young   expected loss=0.2066116  P(node) =0.08211741
# class counts:   192    33    17     0
# probabilities: 0.793 0.136 0.070 0.000 
# 
# Node number 5: 440 observations,    complexity param=0.02986425
# predicted class=Mature  expected loss=0.6136364  P(node) =0.1493044
# class counts:   114   151   170     5
# probabilities: 0.259 0.343 0.386 0.011 
# left son=10 (322 obs) right son=11 (118 obs)
# Primary splits:
#   Sex            splits as  RLR,         improve=30.125160, (0 missing)
# Height         < 0.1025  to the left,  improve=11.624350, (0 missing)
# Shell_weight   < 0.09175 to the left,  improve=10.753220, (0 missing)
# Diameter       < 0.2775  to the left,  improve= 4.834994, (0 missing)
# Viscera_weight < 0.04225 to the left,  improve= 4.390376, (0 missing)
# Surrogate splits:
#   Length         < 0.4725  to the left,  agree=0.734, adj=0.008, (0 split)
# Shucked_weight < 0.062   to the right, agree=0.734, adj=0.008, (0 split)
# 
# Node number 6: 492 observations
# predicted class=Mature  expected loss=0.245935  P(node) =0.1669494
# class counts:    17    71   371    33
# probabilities: 0.035 0.144 0.754 0.067 
# 
# Node number 7: 1773 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2758037  P(node) =0.6016288
# class counts:     6    20  1284   463
# probabilities: 0.003 0.011 0.724 0.261 
# left son=14 (1229 obs) right son=15 (544 obs)
# Primary splits:
#   Shell_weight   < 0.35975 to the left,  improve=42.67379, (0 missing)
# Height         < 0.1725  to the left,  improve=23.04216, (0 missing)
# Shucked_weight < 0.44475 to the right, improve=21.73438, (0 missing)
# Diameter       < 0.4875  to the left,  improve=13.18220, (0 missing)
# Length         < 0.5375  to the right, improve=12.47624, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.344   to the left,  agree=0.895, adj=0.656, (0 split)
# Diameter       < 0.4975  to the left,  agree=0.873, adj=0.585, (0 split)
# Length         < 0.6325  to the left,  agree=0.856, adj=0.531, (0 split)
# Height         < 0.1725  to the left,  agree=0.841, adj=0.482, (0 split)
# Viscera_weight < 0.30175 to the left,  agree=0.836, adj=0.465, (0 split)
# 
# Node number 10: 322 observations
# predicted class=Adult   expected loss=0.6086957  P(node) =0.1092637
# class counts:   102   126    93     1
# probabilities: 0.317 0.391 0.289 0.003 
# 
# Node number 11: 118 observations
# predicted class=Mature  expected loss=0.3474576  P(node) =0.04004072
# class counts:    12    25    77     4
# probabilities: 0.102 0.212 0.653 0.034 
# 
# Node number 14: 1229 observations
# predicted class=Mature  expected loss=0.2229455  P(node) =0.4170343
# class counts:     6    20   955   248
# probabilities: 0.005 0.016 0.777 0.202 
# 
# Node number 15: 544 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.3952206  P(node) =0.1845945
# class counts:     0     0   329   215
# probabilities: 0.000 0.000 0.605 0.395 
# left son=30 (398 obs) right son=31 (146 obs)
# Primary splits:
#   Shucked_weight < 0.53525 to the right, improve=36.48604, (0 missing)
# Viscera_weight < 0.28725 to the right, improve=20.99502, (0 missing)
# Whole_weight   < 1.19525 to the right, improve=15.90028, (0 missing)
# Length         < 0.6225  to the right, improve=11.40289, (0 missing)
# Diameter       < 0.4725  to the right, improve=10.59031, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.32875 to the right, agree=0.888, adj=0.582, (0 split)
# Length         < 0.6325  to the right, agree=0.846, adj=0.425, (0 split)
# Viscera_weight < 0.26375 to the right, agree=0.840, adj=0.404, (0 split)
# Diameter       < 0.4925  to the right, agree=0.807, adj=0.281, (0 split)
# Height         < 0.1575  to the right, agree=0.752, adj=0.075, (0 split)
# 
# Node number 30: 398 observations,    complexity param=0.01312217
# predicted class=Mature  expected loss=0.2864322  P(node) =0.1350526
# class counts:     0     0   284   114
# probabilities: 0.000 0.000 0.714 0.286 
# left son=60 (227 obs) right son=61 (171 obs)
# Primary splits:
#   Shell_weight   < 0.45275 to the left,  improve=18.356970, (0 missing)
# Shucked_weight < 0.63075 to the right, improve= 6.122971, (0 missing)
# Height         < 0.1975  to the left,  improve= 5.531357, (0 missing)
# Whole_weight   < 1.483   to the left,  improve= 4.764440, (0 missing)
# Diameter       < 0.5325  to the left,  improve= 3.729644, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.68675 to the left,  agree=0.812, adj=0.561, (0 split)
# Diameter       < 0.5325  to the left,  agree=0.784, adj=0.497, (0 split)
# Length         < 0.6725  to the left,  agree=0.739, adj=0.392, (0 split)
# Viscera_weight < 0.385   to the left,  agree=0.709, adj=0.322, (0 split)
# Shucked_weight < 0.764   to the left,  agree=0.686, adj=0.269, (0 split)
# 
# Node number 31: 146 observations
# predicted class=Old     expected loss=0.3082192  P(node) =0.04954191
# class counts:     0     0    45   101
# probabilities: 0.000 0.000 0.308 0.692 
# 
# Node number 60: 227 observations
# predicted class=Mature  expected loss=0.1674009  P(node) =0.07702749
# class counts:     0     0   189    38
# probabilities: 0.000 0.000 0.833 0.167 
# 
# Node number 61: 171 observations,    complexity param=0.01312217
# predicted class=Mature  expected loss=0.4444444  P(node) =0.05802511
# class counts:     0     0    95    76
# probabilities: 0.000 0.000 0.556 0.444 
# left son=122 (104 obs) right son=123 (67 obs)
# Primary splits:
#   Shucked_weight < 0.708   to the right, improve=16.938820, (0 missing)
# Length         < 0.7025  to the right, improve= 6.456648, (0 missing)
# Whole_weight   < 1.98275 to the right, improve= 3.783777, (0 missing)
# Viscera_weight < 0.37475 to the right, improve= 3.703621, (0 missing)
# Shell_weight   < 0.568   to the left,  improve= 3.458303, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.7355  to the right, agree=0.813, adj=0.522, (0 split)
# Viscera_weight < 0.351   to the right, agree=0.760, adj=0.388, (0 split)
# Length         < 0.6725  to the right, agree=0.737, adj=0.328, (0 split)
# Diameter       < 0.5375  to the right, agree=0.731, adj=0.313, (0 split)
# Height         < 0.1825  to the right, agree=0.678, adj=0.179, (0 split)
# 
# Node number 122: 104 observations
# predicted class=Mature  expected loss=0.2692308  P(node) =0.03529013
# class counts:     0     0    76    28
# probabilities: 0.000 0.000 0.731 0.269 
# 
# Node number 123: 67 observations
# predicted class=Old     expected loss=0.2835821  P(node) =0.02273498
# class counts:     0     0    19    48
# probabilities: 0.000 0.000 0.284 0.716
rpart.plot(pRp_abalone_model_new, main="Classification for Abalone")

# we change maxDepth to 5 and mibucket to 50
rpControl = rpart.control(minbucket = 60, maxDepth = 5);
pRp_abalone_model_new <- rpart(myFormula,  method = "class", data = trainData, control = rpControl, parms = list(split = "information"))
pRp_abalone_model_new<- prune(pRp_abalone_model_new, cp = pRp_abalone_model_new$cptable[which.min(pRp_abalone_model_new$cptable[,"xerror"]),"CP"])
printcp(pRp_abalone_model_new)
# Classification tree:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# 
# Variables actually used in tree construction:
#   [1] Sex            Shell_weight   Shucked_weight
# 
# Root node error: 1105/2947 = 0.37496
# 
# n= 2947 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.107692      0   1.00000 1.00000 0.023783
# 2 0.050679      1   0.89231 0.89321 0.023186
# 3 0.029864      2   0.84163 0.86244 0.022980
# 4 0.016893      3   0.81176 0.83258 0.022765
# 5 0.010000      6   0.76109 0.79729 0.022491
plotcp(pRp_abalone_model_new)
summary(pRp_abalone_model_new)
# Call:
#   rpart(formula = myFormula, data = trainData, method = "class", 
#         parms = list(split = "information"), control = rpControl)
# n= 2947 
# 
# CP nsplit rel error    xerror       xstd
# 1 0.10769231      0 1.0000000 1.0000000 0.02378338
# 2 0.05067873      1 0.8923077 0.8751131 0.02306713
# 3 0.02986425      2 0.8416290 0.8497738 0.02289087
# 4 0.01689291      3 0.8117647 0.8371041 0.02279865
# 5 0.01000000      6 0.7610860 0.7900452 0.02243154
# 
# Variable importance
# Shell_weight   Whole_weight       Diameter         Length Viscera_weight Shucked_weight            Sex 
# 20             17             16             16             15             14              1 
# 
# Node number 1: 2947 observations,    complexity param=0.1076923
# predicted class=Mature  expected loss=0.3749576  P(node) =1
# class counts:   329   275  1842   501
# probabilities: 0.112 0.093 0.625 0.170 
# left son=2 (682 obs) right son=3 (2265 obs)
# Primary splits:
#   Shell_weight   < 0.11975 to the left,  improve=703.4058, (0 missing)
# Diameter       < 0.3425  to the left,  improve=642.2094, (0 missing)
# Whole_weight   < 0.42325 to the left,  improve=641.5147, (0 missing)
# Length         < 0.4525  to the left,  improve=619.1732, (0 missing)
# Viscera_weight < 0.11025 to the left,  improve=608.0687, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.39425 to the left,  agree=0.970, adj=0.870, (0 split)
# Diameter       < 0.3375  to the left,  agree=0.968, adj=0.862, (0 split)
# Length         < 0.4375  to the left,  agree=0.962, adj=0.837, (0 split)
# Viscera_weight < 0.08025 to the left,  agree=0.956, adj=0.808, (0 split)
# Shucked_weight < 0.16475 to the left,  agree=0.945, adj=0.762, (0 split)
# 
# Node number 2: 682 observations,    complexity param=0.05067873
# predicted class=Young   expected loss=0.5513196  P(node) =0.2314218
# class counts:   306   184   187     5
# probabilities: 0.449 0.270 0.274 0.007 
# left son=4 (242 obs) right son=5 (440 obs)
# Primary splits:
#   Shell_weight   < 0.0555  to the left,  improve=97.99399, (0 missing)
# Diameter       < 0.2225  to the left,  improve=93.53946, (0 missing)
# Whole_weight   < 0.16975 to the left,  improve=86.78150, (0 missing)
# Viscera_weight < 0.03375 to the left,  improve=85.26963, (0 missing)
# Length         < 0.2875  to the left,  improve=84.73763, (0 missing)
# Surrogate splits:
#   Diameter       < 0.2525  to the left,  agree=0.947, adj=0.851, (0 split)
# Whole_weight   < 0.1805  to the left,  agree=0.944, adj=0.843, (0 split)
# Length         < 0.3425  to the left,  agree=0.931, adj=0.806, (0 split)
# Shucked_weight < 0.06575 to the left,  agree=0.906, adj=0.736, (0 split)
# Viscera_weight < 0.03425 to the left,  agree=0.902, adj=0.723, (0 split)
# 
# Node number 3: 2265 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2693157  P(node) =0.7685782
# class counts:    23    91  1655   496
# probabilities: 0.010 0.040 0.731 0.219 
# left son=6 (492 obs) right son=7 (1773 obs)
# Primary splits:
#   Shell_weight   < 0.18975 to the left,  improve=122.30550, (0 missing)
# Height         < 0.1525  to the left,  improve= 76.71986, (0 missing)
# Whole_weight   < 0.80475 to the left,  improve= 73.19802, (0 missing)
# Diameter       < 0.4125  to the left,  improve= 72.27077, (0 missing)
# Viscera_weight < 0.16325 to the left,  improve= 68.90266, (0 missing)
# Surrogate splits:
#   Whole_weight   < 0.65975 to the left,  agree=0.932, adj=0.687, (0 split)
# Diameter       < 0.3975  to the left,  agree=0.919, adj=0.626, (0 split)
# Viscera_weight < 0.13625 to the left,  agree=0.906, adj=0.565, (0 split)
# Length         < 0.5175  to the left,  agree=0.900, adj=0.539, (0 split)
# Shucked_weight < 0.25575 to the left,  agree=0.880, adj=0.449, (0 split)
# 
# Node number 4: 242 observations
# predicted class=Young   expected loss=0.2066116  P(node) =0.08211741
# class counts:   192    33    17     0
# probabilities: 0.793 0.136 0.070 0.000 
# 
# Node number 5: 440 observations,    complexity param=0.02986425
# predicted class=Mature  expected loss=0.6136364  P(node) =0.1493044
# class counts:   114   151   170     5
# probabilities: 0.259 0.343 0.386 0.011 
# left son=10 (322 obs) right son=11 (118 obs)
# Primary splits:
#   Sex            splits as  RLR,         improve=30.125160, (0 missing)
# Height         < 0.1025  to the left,  improve=11.624350, (0 missing)
# Shell_weight   < 0.09175 to the left,  improve=10.753220, (0 missing)
# Diameter       < 0.2775  to the left,  improve= 4.834994, (0 missing)
# Viscera_weight < 0.04225 to the left,  improve= 4.390376, (0 missing)
# Surrogate splits:
#   Length         < 0.4725  to the left,  agree=0.734, adj=0.008, (0 split)
# Shucked_weight < 0.062   to the right, agree=0.734, adj=0.008, (0 split)
# 
# Node number 6: 492 observations
# predicted class=Mature  expected loss=0.245935  P(node) =0.1669494
# class counts:    17    71   371    33
# probabilities: 0.035 0.144 0.754 0.067 
# 
# Node number 7: 1773 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.2758037  P(node) =0.6016288
# class counts:     6    20  1284   463
# probabilities: 0.003 0.011 0.724 0.261 
# left son=14 (1229 obs) right son=15 (544 obs)
# Primary splits:
#   Shell_weight   < 0.35975 to the left,  improve=42.67379, (0 missing)
# Height         < 0.1725  to the left,  improve=23.04216, (0 missing)
# Shucked_weight < 0.44475 to the right, improve=21.73438, (0 missing)
# Diameter       < 0.4875  to the left,  improve=13.18220, (0 missing)
# Length         < 0.5375  to the right, improve=12.47624, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.344   to the left,  agree=0.895, adj=0.656, (0 split)
# Diameter       < 0.4975  to the left,  agree=0.873, adj=0.585, (0 split)
# Length         < 0.6325  to the left,  agree=0.856, adj=0.531, (0 split)
# Height         < 0.1725  to the left,  agree=0.841, adj=0.482, (0 split)
# Viscera_weight < 0.30175 to the left,  agree=0.836, adj=0.465, (0 split)
# 
# Node number 10: 322 observations
# predicted class=Adult   expected loss=0.6086957  P(node) =0.1092637
# class counts:   102   126    93     1
# probabilities: 0.317 0.391 0.289 0.003 
# 
# Node number 11: 118 observations
# predicted class=Mature  expected loss=0.3474576  P(node) =0.04004072
# class counts:    12    25    77     4
# probabilities: 0.102 0.212 0.653 0.034 
# 
# Node number 14: 1229 observations
# predicted class=Mature  expected loss=0.2229455  P(node) =0.4170343
# class counts:     6    20   955   248
# probabilities: 0.005 0.016 0.777 0.202 
# 
# Node number 15: 544 observations,    complexity param=0.01689291
# predicted class=Mature  expected loss=0.3952206  P(node) =0.1845945
# class counts:     0     0   329   215
# probabilities: 0.000 0.000 0.605 0.395 
# left son=30 (398 obs) right son=31 (146 obs)
# Primary splits:
#   Shucked_weight < 0.53525 to the right, improve=36.48604, (0 missing)
# Viscera_weight < 0.28725 to the right, improve=20.99502, (0 missing)
# Whole_weight   < 1.19525 to the right, improve=15.90028, (0 missing)
# Length         < 0.6225  to the right, improve=11.40289, (0 missing)
# Diameter       < 0.4875  to the right, improve=10.14492, (0 missing)
# Surrogate splits:
#   Whole_weight   < 1.32875 to the right, agree=0.888, adj=0.582, (0 split)
# Length         < 0.6325  to the right, agree=0.846, adj=0.425, (0 split)
# Viscera_weight < 0.26375 to the right, agree=0.840, adj=0.404, (0 split)
# Diameter       < 0.4925  to the right, agree=0.807, adj=0.281, (0 split)
# Height         < 0.1575  to the right, agree=0.752, adj=0.075, (0 split)
# 
# Node number 30: 398 observations
# predicted class=Mature  expected loss=0.2864322  P(node) =0.1350526
# class counts:     0     0   284   114
# probabilities: 0.000 0.000 0.714 0.286 
# 
# Node number 31: 146 observations
# predicted class=Old     expected loss=0.3082192  P(node) =0.04954191
# class counts:     0     0    45   101
# probabilities: 0.000 0.000 0.308 0.692 
rpart.plot(pRp_abalone_model_new, main="Classification for Abalone")

# *********************************************** CONCLUSION **************************************************************
# My exploration and analysis was largely guided by the use of histograms, 
# helping me delineate the spectrum of abalone ages into four main classifications - Young, Adult, Mature and Old 
# The initial findings delivered a satisfactory degree of precision. Nonetheless, I embarked on a journey to fine-tune 
# these results.
# 
# A loss matrix is a tool used in classification tasks to assign costs or penalties to different types of prediction
# errors made by a machine learning model 
# However in my situation it unintentionally worsened the extent of the errors.
# 
# I implemented tree pruning as a measure to simplify our decision tree for classifying ages in the abalone dataset.
# However, this led to a more streamlined tree, but it also resulted in an increase in the error ratio, 
# particularly in recall. Adjusting maxDepth showed little to no effect, while reducing minbucket sparked the expansion of the
# tree, alongside a decline in the error rate, especially in recall. From this, we glean that decision tree construction 
# involves a delicate balance between precision and recall when classifying ages in the abalone dataset. Depending on 
# individual requirements, a client can tilt the balance towards either aspect or work towards striking an equilibrium between
# the two.
# Equipped with this classifier, an analyst can predict the age of abalone based on various attributes such as diameter, 
# height, and others. The accuracy of these predictions is crucial for tasks such as stock assessment and resource management. 
# Misclassification of abalone age can lead to inaccurate assessments, thereby impacting decision-making processes. 
# Underestimating the age of mature abalone can result in unsustainable harvesting practices, whereas overestimating the age 
# of juvenile abalone may lead to overly conservative management strategies.

