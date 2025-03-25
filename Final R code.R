# Install the required package if not already installed
install.packages("nnet")
install.packages("caret")
# Load necessary libraries
library(nnet)
install.packages("readxl")  
library(readxl)
# Read the dataset (update the path to your file)
setwd("D:/Data Science/Semester 4/Project")
data <- read_excel("Input parameters.xlsx")
# Examine the dataset
head(data)
colnames(data)
colnames(data)[2] <- "Infill_Density"
colnames(data)[3] <- "Layer_Thickness"
colnames(data)[4] <- "Raster_Angle"
colnames(data)[5] <- "Tensile_Strength"
# Normalize the input features (Infill Density, Layer Thickness, Raster Angle)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data$Infill_Density <- normalize(data$Infill_Density)
data$Layer_Thickness <- normalize(data$Layer_Thickness)
data$Raster_Angle <- normalize(data$Raster_Angle)
# Split data into training and testing sets (80% training, 20% testing)
set.seed(123)  # For reproducibility
sample_size <- floor(0.8 * nrow(data))
train_indices <- sample(seq_len(nrow(data)), size = sample_size)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
# View the normalized and split dataset
head(train_data)
library(caret)
train_control <- trainControl(method = "cv", number = 10)
tune_grid <- expand.grid(
  .size = c(10, 15, 20,1000), 
  .decay = c(0.01, 0.05, 0.1,0.5)
)
nn_cv_model <- train(
  Tensile_Strength ~ Infill_Density + Layer_Thickness + Raster_Angle,
  data = train_data,
  method = "nnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  linout = TRUE,
  maxit = 10000
)
# View the best model and performance metrics
nn_cv_model$bestTune
nn_cv_model$results
# Plot cross-validation results for tuning parameters
plot(nn_cv_model)




# Generate predictions using the trained model
predictions <- predict(nn_cv_model, newdata = test_data)

# Plot Actual vs. Predicted values
plot(test_data$Tensile_Strength, predictions, 
     main = "Actual vs. Predicted Tensile Strength",
     xlab = "Actual Tensile Strength (MPa)", 
     ylab = "Predicted Tensile Strength (MPa)",
     pch = 19, col = "blue")

# Add a reference line for perfect prediction
abline(0, 1, col = "red")

# Calculate residuals
residuals <- test_data$Tensile_Strength - predictions

# Plot residuals
plot(predictions, residuals,
     main = "Residual Plot",
     xlab = "Predicted Tensile Strength (MPa)", 
     ylab = "Residuals",
     pch = 19, col = "darkgreen")

# Add a horizontal line at 0
abline(h = 0, col = "red")

# Calculate performance metrics
mse <- mean((predictions - test_data$Tensile_Strength)^2)
mae <- mean(abs(predictions - test_data$Tensile_Strength))
r_squared <- cor(predictions, test_data$Tensile_Strength)^2

# Print out metrics
cat("Model Performance Metrics:\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared:", r_squared, "\n")

# Plot cross-validation results (RMSE over different sizes and decays)
plot(nn_cv_model, metric = "RMSE")

# Histogram of prediction errors
hist(residuals, 
     main = "Distribution of Prediction Errors",
     xlab = "Residuals (Actual - Predicted)",
     col = "lightblue", breaks = 10)


# Generate predictions using the trained model on the training data
train_predictions <- predict(nn_cv_model, newdata = train_data)

# Calculate the error (Actual - Predicted)
error <- train_data$Tensile_Strength - train_predictions
# Create the data frame in the specified format
predicted_vs_actual <- data.frame(
  `SAMPLE NO.` = 1:nrow(train_data),
  `ACTUAL TENSILE STRENGTH (N/mm²)` = train_data$Tensile_Strength,
  `PREDICTED TENSILE STRENGTH (N/mm²)` = train_predictions,
  `ERROR (N/mm²)` = error
)
# Specify the output file path
output_file <- "Predicted_vs_Actual_Tensile_Strength_Format.csv"

# Write the data frame to a CSV file
write.csv(predicted_vs_actual, output_file, row.names = FALSE)
# Confirmation message
cat("Predicted vs Actual values saved in the specified format to", output_file)






