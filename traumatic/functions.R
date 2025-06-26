# functions for data thizerry bege

cp.select <- function(big.tree) {
	min.x <- which.min(big.tree$cptable[, 4]) #column 4 is xerror
	for(i in 1:nrow(big.tree$cptable)) {
	    if(big.tree$cptable[i, 4] < big.tree$cptable[min.x, 4] + big.tree$cptable[min.x, 5]) return(big.tree$cptable[i, 1]) #column 5: xstd, column 1: cp 
  	}
}


# Define your objective function
obj_fun_nodesize = function(x) {
  selected_vars3 <- c(dependent_variable, top_3_vars)
  
  if (!exists("dependent_variable") || !exists("top_3_vars")) {
    stop("Error: dependent_variable or top_3_vars is not defined!")
  }
  
  rf_model_nodesize <- randomForest(as.formula(paste(dependent_variable, "~ .")),
                                    data = train_data_top3,
                                    nodesize = as.integer(x$nodesize))  # Use nodesize instead
  
  predictions <- predict(rf_model_nodesize, newdata = test_data_top3)
  conf_matrix <- confusionMatrix(predictions, test_data_top3[[dependent_variable]])
  
  # Return Recall (Sensitivity)
  return(as.numeric(conf_matrix$byClass['Sensitivity']))
}


# Define a function to compute learning curves
compute_learning_curves <- function(train_data, test_data, dependent_variable, top_3_vars, best_nodesize) {
  # Initialize storage for results
  learning_curve_results <- data.frame(
    Training_Size = integer(),
    Accuracy = numeric(),
    Precision = numeric(),
    Recall = numeric(),
    Specificity = numeric(),
    F1 = numeric(),
    AUC = numeric()
  )
 # Define training sizes (e.g., percentages of total training data)
  training_sizes <- seq(0.1, 1.0, by = 0.1) # From 10% to 100% of the training data
  
  for (size in training_sizes) {
    # Subset the training data
    subset_size <- floor(size * nrow(train_data))
    train_subset <- train_data[1:subset_size, ]
    
    # Select top variables
    selected_vars3 <- c(dependent_variable, top_3_vars)
    train_subset_top3 <- train_subset[, selected_vars3]
    test_data_top3 <- test_data[, selected_vars3]
    
    # Train the Random Forest model with optimized nodesize
    rf_model <- randomForest(
      as.formula(paste(dependent_variable, "~ .")),
      data = train_subset_top3,
      ntree = 500,                     # Fixed number of trees
      mtry = sqrt(length(top_3_vars)), # Default mtry for classification
      nodesize = best_nodesize         # Optimized nodesize
    )
    
    # Make predictions
    rf_probabilities <- predict(rf_model, newdata = test_data_top3, type = "prob")[, 2]
    predictions <- predict(rf_model, newdata = test_data_top3)
    
    # Compute confusion matrix and metrics
    conf_matrix <- confusionMatrix(predictions, test_data_top3[[dependent_variable]])
    
    metrics <- c(
      Accuracy = conf_matrix$overall['Accuracy'],
      Precision = conf_matrix$byClass['Pos Pred Value'],
      Recall = conf_matrix$byClass['Sensitivity'],
      Specificity = conf_matrix$byClass['Specificity'],
      F1 = conf_matrix$byClass['F1'],
      AUC = auc(roc(test_data_top3[[dependent_variable]], rf_probabilities))
    )
    
    # Store results
    learning_curve_results <- rbind(
      learning_curve_results,
      setNames(c(size * 100, metrics), c("Training_Size", "Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"))
    )
  }
  
  return(learning_curve_results)
}

compute_learning_curves2 <- function(train_data, test_data, dependent_variable, top_3_vars, best_max_depth) {
  # Initialize storage for results
  learning_curve_results <- data.frame(
    Training_Size = numeric(),
    Accuracy = numeric(),
    Precision = numeric(),
    Recall = numeric(),
    Specificity = numeric(),
    F1 = numeric(),
    AUC = numeric()
  )
    # Define training sizes (e.g., percentages of total training data)
  training_sizes <- seq(0.1, 1.0, by = 0.1)  # From 10% to 100% of the training data
  
  for (size in training_sizes) {
    # Subset the training data
    subset_size <- floor(size * nrow(train_data))
    train_subset <- train_data[sample(nrow(train_data), subset_size), ]
    
    # Select relevant features
    selected_vars3 <- c(dependent_variable, top_3_vars)
    train_subset_top3 <- train_subset[, selected_vars3, drop = FALSE]
    test_data_top3 <- test_data[, selected_vars3, drop = FALSE]
    
    # Train the Random Forest model using ranger
    rf_model <- ranger(
      formula = as.formula(paste(dependent_variable, "~ .")),
      data = train_subset_top3,
      max.depth = best_max_depth,  # Optimized max.depth parameter
      probability = TRUE
    )
    
    # Get probabilities for class "1"
    rf_probabilities <- predict(rf_model, data = test_data_top3)$predictions[, 2]
    
    # Convert probabilities to class predictions
    predictions <- ifelse(rf_probabilities > 0.5, 1, 0)
    
    # Convert to factor with correct levels
    predictions <- factor(predictions, levels = levels(test_data_top3[[dependent_variable]]))
    
    # Compute confusion matrix and metrics
    conf_matrix <- confusionMatrix(predictions, test_data_top3[[dependent_variable]])
    
    # Compute AUC
    auc_value <- auc(roc(test_data_top3[[dependent_variable]], rf_probabilities))
    
    # Store results
    learning_curve_results <- rbind(
      learning_curve_results,
      data.frame(
        Training_Size = size * 100,
        Accuracy = conf_matrix$overall['Accuracy'],
        Precision = conf_matrix$byClass['Pos Pred Value'],
        Recall = conf_matrix$byClass['Sensitivity'],
        Specificity = conf_matrix$byClass['Specificity'],
        F1 = conf_matrix$byClass['F1'],
        AUC = auc_value
      )
    )
  }
  
  return(learning_curve_results)
}


obj_fun_max_depth = function(x) {
  print(x)
  
  # Validate inputs
  if (is.null(dependent_variable) || is.null(top_3_vars) || length(top_3_vars) == 0) {
    stop("Error: dependent_variable or top_3_vars is not defined or empty!")
  }
  
  # Select relevant features
  selected_vars3 <- c(dependent_variable, top_3_vars)
  train_data_top3 <- train_data_top3[selected_vars3]
  test_data_top3 <- test_data_top3[selected_vars3]
  
  # Check for missing values
  if (sum(is.na(train_data_top3)) > 0 || sum(is.na(test_data_top3)) > 0) {
    stop("Error: Missing values detected in train_data_top3 or test_data_top3!")
  }
  
  # Train model
  rf_model_max_depth <- ranger(
    dependent.variable.name = dependent_variable,
    data = train_data_top3,
    max.depth = as.integer(x$max.depth),
    probability = TRUE
  )
  
  # Get predictions (probabilities)
  predictions_max_depth <- predict(rf_model_max_depth, data = test_data_top3)$predictions
  
  # Convert probabilities to class labels
  class_labels <- colnames(predictions_max_depth)  # Ensure correct class names
  predictions_max_depth <- apply(predictions_max_depth, 1, function(x) class_labels[which.max(x)])
  
  # Ensure factor levels match
  predictions_max_depth <- factor(predictions_max_depth, levels = levels(test_data_top3[[dependent_variable]]))
  
  # Compute confusion matrix
  conf_matrix_max_depth <- confusionMatrix(predictions_max_depth, test_data_top3[[dependent_variable]])
  
  # Return Recall (Sensitivity)
  recall <- as.numeric(conf_matrix_max_depth$byClass['Sensitivity'])
  return(recall + runif(1, -0.001, 0.001))  # Add slight noise to avoid ties
}

obj_fun_nodesize = function(x) {
  selected_vars3 <- c(dependent_variable, top_3_vars)
  
  if (!exists("dependent_variable") || !exists("top_3_vars")) {
    stop("Error: dependent_variable or top_3_vars is not defined!")
  }
  
  rf_model_nodesize <- randomForest(as.formula(paste(dependent_variable, "~ .")),
                                    data = train_data_top3,
                                    nodesize = as.integer(x$nodesize))  # Use nodesize instead
  
  predictions <- predict(rf_model_nodesize, newdata = test_data_top3)
  conf_matrix <- confusionMatrix(predictions, test_data_top3[[dependent_variable]])
  
  # Return Recall (Sensitivity)
  return(as.numeric(conf_matrix$byClass['Sensitivity']))
}


##OPTIMISATION
obj_fun_nodesize <- function(x) {
  # Replace with your model training + evaluation logic
  # For example:
  nodesize_val <- x$nodesize
  recall_val <- runif(1)  # Dummy random recall — replace this
  return(recall_val)
}
























