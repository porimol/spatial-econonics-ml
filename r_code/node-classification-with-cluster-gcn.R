library(tidyverse)
library(igraph)
library(caret)
library(scales)

# Load dataset
df <- read_csv("dataset/migration-edges.csv")

# Encode node names to integers
nodes <- unique(c(df$Source, df$Target))
node_map <- setNames(0:(length(nodes)-1), nodes)
df <- df %>%
  mutate(Source = node_map[Source],
         Target = node_map[Target])

# Rename column
colnames(df)[colnames(df) == "target_country_gdp"] <- "GDP"

# Get last known GDP per node
gdp_df <- df %>%
  group_by(Target) %>%
  summarise(GDP = last(GDP))

node_features <- tibble(Node = 0:max(df$Source)) %>%
  left_join(gdp_df, by = c("Node" = "Target")) %>%
  mutate(GDP = replace_na(GDP, 0))

# Normalize GDP
node_features$GDP <- rescale(node_features$GDP, to = c(0, 1))
head(node_features)

# Prepare edge list and edge attribute
edges <- df %>% select(Source, Target)
edge_attr <- as.numeric(df$Migrated)
edge_attr[is.na(edge_attr)] <- 0



# Build Graph and Label Nodes
# Create igraph object
g <- graph_from_data_frame(edges, directed = TRUE)

# Calculate in-degree and define hubs
in_deg <- degree(g, mode = "in")
threshold <- quantile(in_deg, 0.9)
labels <- ifelse(in_deg >= threshold, 1, 0)

# Use MLP Instead of GCN in R (no torch_geometric in R)
install.packages("torch")
torch::install_torch()
library(torch)

x <- torch_tensor(matrix(node_features$GDP, ncol = 1), dtype = torch_float())
y <- torch_tensor(labels, dtype = torch_long())

# Define a simple MLP model
model <- nn_module(
  "MLP",
  initialize = function() {
    self$fc1 <- nn_linear(1, 32)
    self$fc2 <- nn_linear(32, 16)
    self$fc3 <- nn_linear(16, 2)
  },
  forward = function(x) {
    x %>% self$fc1() %>% nnf_relu() %>%
      self$fc2() %>% nnf_relu() %>%
      self$fc3()
  }
)

net <- model()
optimizer <- optim_adam(net$parameters, lr = 0.01)
loss_fn <- nn_cross_entropy_loss()

y <- torch_tensor(labels + 1, dtype = torch_long())

for (epoch in 1:100) {
  net$train()
  optimizer$zero_grad()
  output <- net(x)
  loss <- loss_fn(output, y)
  loss$backward()
  optimizer$step()
  
  if (epoch %% 10 == 0) {
    pred <- output$argmax(dim = 2)
    acc <- sum((pred == y)$to(dtype = torch_float()))$item() / length(y)
    cat(sprintf("Epoch %d: Loss = %.4f, Accuracy = %.4f\n", epoch, loss$item(), acc))
  }
}



# Embedding & Clustering (Using PCA/TSNE in R)
library(Rtsne)
library(cluster)
library(ggplot2)

# Extract embeddings (use output from hidden layer)
# Forward pass manually
net$eval()  # Set model to evaluation mode
# Step-by-step
h1 <- net$fc1(x)
a1 <- nnf_relu(h1)
h2 <- net$fc2(a1)
embeddings <- h2$detach()
embeddings <- embeddings$to(device = "cpu")
embeddings <- as_array(embeddings)

# t-SNE
# Add small noise to break ties (jitter) — safe if values are continuous
jittered_embeddings <- embeddings + matrix(rnorm(length(embeddings), sd = 1e-6), nrow = nrow(embeddings))
# Then run t-SNE
tsne_out <- Rtsne(jittered_embeddings, dims = 2)$Y

kmeans_res <- kmeans(embeddings, centers = 2)$cluster

ggplot(data.frame(x = tsne_out[,1], y = tsne_out[,2], cluster = factor(kmeans_res)),
       aes(x, y, color = cluster)) +
  geom_point(alpha = 0.8) +
  theme_minimal() +
  ggtitle("t-SNE of Node Embeddings")



# Classification with Logistic Regression and Randoim Forest
install.packages("smotefamily")
library(smotefamily)
library(lightgbm)
library(ROCR)

# Convert to dataframe for SMOTE
smote_df <- data.frame(embeddings)
smote_df$label <- as.factor(labels)

# SMOTE oversampling
smote_result <- SMOTE(X = smote_df[, -ncol(smote_df)], target = smote_df$label, K = 5)
X_resampled <- smote_result$data[, -ncol(smote_result$data)]
y_resampled <- smote_result$data$class

# Train/test split
train_idx <- createDataPartition(y_resampled, p = 0.8, list = FALSE)
X_train <- X_resampled[train_idx, ]
X_test <- X_resampled[-train_idx, ]
y_train <- y_resampled[train_idx]
y_test <- y_resampled[-train_idx]

# Combine features and labels into data frames
train <- data.frame(X_train)
train$label <- as.factor(y_train)
test <- data.frame(X_test)
test$label <- as.factor(y_test)

# Logistic Regression
log_model <- glm(label ~ ., data = train, family = binomial)
# Predict probabilities on test set
log_preds <- predict(log_model, test, type = "response")
# Convert probabilities to class predictions
log_class <- ifelse(log_preds > 0.5, 1, 0)
# Calculate accuracy
log_acc <- mean(log_class == as.numeric(test$label) - 1)
cat(sprintf("Logistic Regression Accuracy: %.2f%%\n", log_acc * 100))


install.packages("randomForest")
library(randomForest)

# Train the model
rf_model <- randomForest(label ~ ., data = train, ntree = 100, mtry = 3, importance = TRUE)
# Predict on test data
rf_pred <- predict(rf_model, newdata = test)
# Accuracy
rf_acc <- mean(rf_pred == test$label)
cat(sprintf("Random Forest Accuracy: %.2f%%\n", rf_acc * 100))

importance(rf_model)
varImpPlot(rf_model)

