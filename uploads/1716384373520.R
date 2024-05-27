#EDAMI LAB 3 - Ewen Liam
#Clustering

# Import libraries
library(dbscan)
library(fpc)
library(cluster)
library(factoextra)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(mclust) 


# Load the datasets
# Load Wine Quality data
wine <- read.csv("~/Lab03/winequality-red.csv", header = TRUE, sep = ";")

# Preprocess wine data
wine_noQuality <- wine %>% select(-quality)

# Data scaling
wine_scaled <- scale(wine_noQuality)

# Function to calculate accuracy
accuracyCalc <- function(confTbl, startCol) {
  corr = 0
  for (i in startCol:ncol(confTbl)) {
    corr = corr + max(confTbl[, i])
  }
  accuracy = corr / sum(confTbl)
  return(accuracy)
}

# Function to calculate silhouette width and corrected Rand index
evaluate_clustering_quality <- function(data, clusters, true_labels) {
  silhouette_score <- mean(silhouette(clusters, dist(data))[, 3])
  rand_index <- adjustedRandIndex(clusters, as.numeric(true_labels))
  return(list(silhouette = silhouette_score, rand = rand_index))
}

# K-means for Wine
wine_kmeans <- function(data, k) {
  results <- kmeans(data, centers = k, nstart = 20)
  return(results)
}

# PAM for Wine
wine_pam <- function(data, k) {
  results <- pam(data, k)
  return(results)
}

# Hierarchical clustering for Wine
wine_hclust <- function(data, k) {
  distM <- dist(data)
  hclust_res <- hclust(distM, method = "average")
  clusters <- cutree(hclust_res, k = k)
  return(clusters)
}

# DBSCAN for Wine
wine_dbscan <- function(data) {
  results <- dbscan(data, eps = 0.5)
  return(results)
}

# Perform clustering and evaluate for Wine
set.seed(7777)
results_wine <- list()

for (i in 1:10) {
  kmeans_res <- wine_kmeans(wine_scaled, 6)
  pam_res <- wine_pam(wine_scaled, 6)
  hclust_res <- wine_hclust(wine_scaled, 6)
  dbscan_res <- wine_dbscan(wine_scaled)
  
  kmeans_eval <- evaluate_clustering_quality(wine_scaled, kmeans_res$cluster, wine$quality)
  pam_eval <- evaluate_clustering_quality(wine_scaled, pam_res$clustering, wine$quality)
  hclust_eval <- evaluate_clustering_quality(wine_scaled, hclust_res, wine$quality)
  dbscan_eval <- evaluate_clustering_quality(wine_scaled, dbscan_res$cluster, wine$quality)
  
  kmeans_acc <- accuracyCalc(table(wine$quality, kmeans_res$cluster), 1)
  pam_acc <- accuracyCalc(table(wine$quality, pam_res$clustering), 1)
  hclust_acc <- accuracyCalc(table(wine$quality, hclust_res), 1)
  dbscan_acc <- if (length(unique(dbscan_res$cluster)) > 1) accuracyCalc(table(wine$quality, dbscan_res$cluster), 1) else NA
  
  results_wine[[i]] <- list(
    kmeans = list(accuracy = kmeans_acc, quality = kmeans_eval),
    pam = list(accuracy = pam_acc, quality = pam_eval),
    hclust = list(accuracy = hclust_acc, quality = hclust_eval),
    dbscan = list(accuracy = dbscan_acc, quality = dbscan_eval)
  )
}

# Summarize results
results_summary <- do.call(rbind, lapply(results_wine, function(x) {
  lapply(x, function(y) c(y$accuracy, y$quality$silhouette, y$quality$rand))
}))
results_summary_df <- as.data.frame(results_summary)

# Print summarized results
print(results_summary_df)

# Visualize clustering results for the first trial
fviz_cluster(wine_kmeans(wine_scaled, 6), data = wine_scaled) + ggtitle("K-means Clustering")
fviz_cluster(wine_pam(wine_scaled, 6), data = wine_scaled) + ggtitle("PAM Clustering")
hclust_res <- wine_hclust(wine_scaled, 6)
plot(hclust_res, labels = FALSE, main = "Hierarchical Clustering Dendrogram")
dbscan_res <- wine_dbscan(wine_scaled)
fviz_cluster(dbscan_res, data = wine_scaled, stand = FALSE, geom = "point") + ggtitle("DBSCAN Clustering")

# Create the contingency table
contingency_table <- table(wine$quality, wine_kmeans(wine_scaled, 6)$cluster)

# Convert the table to a data frame for ggplot2
contingency_df <- as.data.frame(as.table(contingency_table))

# Plot the heatmap
# The heatmap of wine quality vs. cluster assignment 
# visually represents how well the clusters align with the actual wine quality.
ggplot(contingency_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "Heatmap of Wine Quality vs. Cluster Assignment",
       x = "Wine Quality",
       y = "Cluster",
       fill = "Count") +
  theme_minimal()

# Conclusion
# Overall, K-means and PAM clustering methods seem to perform 
# better for this dataset based on the accuracy calculated. 


