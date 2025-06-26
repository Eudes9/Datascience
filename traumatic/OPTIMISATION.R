source("functions.R")
source("prepareData.R")
source("main.R")
source("analyseResultats.R")


set.seed(42)

#Define hyperparameter grid
 nodesize_values <- seq(1, 30, by = 1)
 mtry_values <- c(1, 2, 3)  # Adjust based on the number of features you have
#Create folds
 cv_folds <- createFolds(train_data_top3[[dependent_variable]], k = 10, list = TRUE, returnTrain = TRUE)

#Initialize results container
 results <- data.frame()

# Grid search
for (ns in nodesize_values) {
  for (m in mtry_values) {
    cv_metrics <- data.frame()
    
    for (fold in seq_along(cv_folds)) {
      train_idx <- cv_folds[[fold]]
      cv_train <- train_data_top3[train_idx, ]
      cv_valid <- train_data_top3[-train_idx, ]
      
      rf_model <- randomForest(
        as.formula(paste(dependent_variable, "~ .")),
        data = cv_train,
        nodesize = ns,
        mtry = m,
        importance = FALSE
      )
      
      preds <- predict(rf_model, newdata = cv_valid)
      probs <- predict(rf_model, newdata = cv_valid, type = "prob")[, 2]
      
      cm <- confusionMatrix(preds, cv_valid[[dependent_variable]])
      roc_obj <- roc(cv_valid[[dependent_variable]], probs)
      
      cv_metrics <- rbind(cv_metrics, data.frame(
        Accuracy = cm$overall["Accuracy"],
        Precision = cm$byClass["Pos Pred Value"],
        Recall = cm$byClass["Sensitivity"],
        F1 = cm$byClass["F1"],
        AUC = auc(roc_obj)
      ))
    }
    
    avg_metrics <- colMeans(cv_metrics)
    results <- rbind(results, cbind(nodesize = ns, mtry = m, t(avg_metrics)))
  }
}

# Choose best hyperparameters based on Recall (or another metric)
 best_row <- results[which.max(results$Recall), ]
 print("Best hyperparameters:")
 print(best_row)

# Train final model with best parameters
 final_model <- randomForest(
   as.formula(paste(dependent_variable, "~ .")),
   data = train_data_top3,
   nodesize = best_row$nodesize,
   mtry = best_row$mtry,
   importance = FALSE
 )

# Evaluate on test set
 final_preds <- predict(final_model, newdata = test_data_top3)
 final_probs <- predict(final_model, newdata = test_data_top3, type = "prob")[, 2]

 conf_matrix <- confusionMatrix(final_preds, test_data_top3[[dependent_variable]])
 roc_obj <- roc(test_data_top3[[dependent_variable]], final_probs)

 final_metrics <- c(
   Accuracy = conf_matrix$overall["Accuracy"],
   Precision = conf_matrix$byClass["Pos Pred Value"],
   Recall = conf_matrix$byClass["Sensitivity"],
   Specificity = conf_matrix$byClass["Specificity"],
   F1 = conf_matrix$byClass["F1"],
   AUC = auc(roc_obj)
 )

 print("Final evaluation on test data:")
 print(final_metrics)



#---ALL THE DATA	

# Define hyperparameter grid
nodesize_values <- seq(1, 20, by = 1)
mtry_values <- c(1, 2, 3)  # Adjust based on the number of features you have
# Create folds
cv_folds <- createFolds(train_data_top3[[dependent_variable]], k = 10, list = TRUE, returnTrain = TRUE)

# Initialize results container
results <- data.frame()

# Grid search
for (ns in nodesize_values) {
  for (m in mtry_values) {
    cv_metrics <- data.frame()
    
    for (fold in seq_along(cv_folds)) {
      train_idx <- cv_folds[[fold]]
      cv_train <- train_data[train_idx, ]
      cv_valid <- train_data[-train_idx, ]
      
      rf_model <- randomForest(
        as.formula(paste(dependent_variable, "~ .")),
        data = cv_train,
        nodesize = ns,
        mtry = m,
        importance = FALSE
      )
      
      preds <- predict(rf_model, newdata = cv_valid)
      probs <- predict(rf_model, newdata = cv_valid, type = "prob")[, 2]
      
      cm <- confusionMatrix(preds, cv_valid[[dependent_variable]])
      roc_obj <- roc(cv_valid[[dependent_variable]], probs)
      
      cv_metrics <- rbind(cv_metrics, data.frame(
        Accuracy = cm$overall["Accuracy"],
        Precision = cm$byClass["Pos Pred Value"],
        Recall = cm$byClass["Sensitivity"],
        F1 = cm$byClass["F1"],
        AUC = auc(roc_obj)
      ))
    }
    
    avg_metrics <- colMeans(cv_metrics)
    results <- rbind(results, cbind(nodesize = ns, mtry = m, t(avg_metrics)))
  }
}

# Choose best hyperparameters based on Recall (or another metric)
best_row <- results[which.max(results$Recall), ]
print("Best hyperparameters:")
print(best_row)

# Train final model with best parameters
final_model <- randomForest(
  as.formula(paste(dependent_variable, "~ .")),
  data = train_data,
  nodesize = best_row$nodesize,
  mtry = best_row$mtry,
  importance = FALSE
)

# Evaluate on test set
final_preds <- predict(final_model, newdata = test_data)
final_probs <- predict(final_model, newdata = test_data, type = "prob")[, 2]

conf_matrix <- confusionMatrix(final_preds, test_data[[dependent_variable]])
roc_obj <- roc(test_data[[dependent_variable]], final_probs)

final_metrics <- c(
  Accuracy = conf_matrix$overall["Accuracy"],
  Precision = conf_matrix$byClass["Pos Pred Value"],
  Recall = conf_matrix$byClass["Sensitivity"],
  Specificity = conf_matrix$byClass["Specificity"],
  F1 = conf_matrix$byClass["F1"],
  AUC = auc(roc_obj)
)

print("Final evaluation on test data:")
print(final_metrics)




###############

library(randomForest)
library(pROC)
library(caret)

tune_and_evaluate_rf <- function(train_data, test_data, dependent_variable, 
                                 nodesize_values = seq(1, 20, 1), 
                                 mtry_values = NULL,
                                 cv_folds_k = 10,
                                 metric_to_optimize = "Recall") {
  set.seed(42)
  
  # Déterminer les variables indépendantes
  independent_variables <- setdiff(names(train_data), dependent_variable)
  
  # Par défaut : mtry entre 1 et nb de variables
  if (is.null(mtry_values)) {
    mtry_values <- 1:length(independent_variables)
  }
  
  # Créer les folds
  cv_folds <- createFolds(train_data[[dependent_variable]], k = cv_folds_k, list = TRUE, returnTrain = TRUE)
  
  results <- data.frame()
  
  for (ns in nodesize_values) {
    for (m in mtry_values) {
      cv_metrics <- data.frame()
      
      for (fold in seq_along(cv_folds)) {
        train_idx <- cv_folds[[fold]]
        cv_train <- train_data[train_idx, ]
        cv_valid <- train_data[-train_idx, ]
        
        rf_model <- randomForest(
          as.formula(paste(dependent_variable, "~ .")),
          data = cv_train,
          nodesize = ns,
          mtry = m,
          importance = FALSE
        )
        
        preds <- predict(rf_model, newdata = cv_valid)
        probs <- predict(rf_model, newdata = cv_valid, type = "prob")[, 2]
        
        cm <- confusionMatrix(preds, cv_valid[[dependent_variable]])
        roc_obj <- roc(cv_valid[[dependent_variable]], probs)
        
        cv_metrics <- rbind(cv_metrics, data.frame(
          Accuracy = cm$overall["Accuracy"],
          Precision = cm$byClass["Pos Pred Value"],
          Recall = cm$byClass["Sensitivity"],
          F1 = cm$byClass["F1"],
          AUC = auc(roc_obj)
        ))
      }
      
      avg_metrics <- colMeans(cv_metrics)
      results <- rbind(results, cbind(nodesize = ns, mtry = m, t(avg_metrics)))
    }
  }
  
  # Sélection du meilleur modèle
  best_row <- results[which.max(results[[metric_to_optimize]]), ]
  print("Best hyperparameters:")
  print(best_row)
  
  # Entraîner le modèle final
  final_model <- randomForest(
    as.formula(paste(dependent_variable, "~ .")),
    data = train_data,
    nodesize = best_row$nodesize,
    mtry = best_row$mtry,
    importance = TRUE
  )
  
  final_preds <- predict(final_model, newdata = test_data)
  final_probs <- predict(final_model, newdata = test_data, type = "prob")[, 2]
  
  conf_matrix <- confusionMatrix(final_preds, test_data[[dependent_variable]])
  roc_obj <- roc(test_data[[dependent_variable]], final_probs)
  
  final_metrics <- c(
    Accuracy = conf_matrix$overall["Accuracy"],
    Precision = conf_matrix$byClass["Pos Pred Value"],
    Recall = conf_matrix$byClass["Sensitivity"],
    Specificity = conf_matrix$byClass["Specificity"],
    F1 = conf_matrix$byClass["F1"],
    AUC = auc(roc_obj)
  )
  
  print("Final evaluation on test data:")
  print(final_metrics)
  
  return(list(
    best_hyperparams = best_row,
    final_metrics = final_metrics,
    model = final_model,
    cv_results = results
  ))
}

result_top3 <- tune_and_evaluate_rf(
  train_data = train_data_top3,
  test_data = test_data_top3,
  dependent_variable = dependent_variable
)


result_all <- tune_and_evaluate_rf(
  train_data = train_data,
  test_data = test_data,
  dependent_variable = dependent_variable
)









