rm(list = ls())
# pdf('Faget_amélioré.pdf')
# Charger les bibliothèques nécessaires
library(ggplot2)
library(gridExtra)
library(e1071)
library(randomForest)
library(rpart)
#library(rpart.plot)
library(caret)
library(reshape2)
library(dplyr)         # For the pipe operator `%>%`
library(tibble)        # For rownames_to_column()
library(mlrMBO)  #  Bayesian Optimisation ????
library(smoof)  # Required for makeSingleObjectiveFunction
library(lhs)     # Required for Latin Hypercube Sampling
library(rgenoud)
library(ranger)
library(mlr)
library(ParamHelpers)
library(DiceKriging)
library(pROC)  # Load the package
library(xtable) # for latex
library(caret)
library(pROC)

setwd("C:/Users/jeane/Documents/traumatic")

# Model without score Faget et Bips 
source("functions.R")
source("prepareData.R")


#MODELES
#	set.seed(123)
	n_iterations = 100


#Initialize data frames to store results
	svm_metrics = rf_metrics = dt_metrics = rf_top4_metrics = rf_top3_metrics = data.frame(Accuracy = numeric(n_iterations), Precision = numeric(n_iterations),
                          Sensitivity = numeric(n_iterations), Specificity = numeric(n_iterations),
                          F1 = numeric(n_iterations), AUC = numeric(n_iterations))

	#Store confusion matrices for each iteration
	svm_conf_matrices <- list()
	rf_conf_matrices <- list()
	dt_conf_matrices <- list()
# Loop for learning all models over various samples	
	for (i in 1:n_iterations) {
		cat("\nIteration:", i, "\n")  # Track iteration progress
 #Split the data
		train_index <- sample(1:nrow(data2), 0.7 * nrow(data2))
		train_data <- data2[train_index, ]
		test_data <- data2[-train_index, ]
#Remove single-level variables
		single_level_vars <- sapply(train_data, function(x) length(unique(x)) == 1)
		train_data <- train_data[, !single_level_vars, drop = FALSE]
		test_data <- test_data[, !single_level_vars, drop = FALSE]
	  
# ---- SVM Model ----
		svm_model <- svm(as.formula(paste(dependent_variable, "~ .")), 
	                   data = train_data, kernel = "radial", probability = TRUE)
		svm_predictions <- predict(svm_model, newdata = test_data)
		svm_predictions <- factor(svm_predictions, levels = levels(test_data[[dependent_variable]]))
		svm_conf_matrix <- confusionMatrix(svm_predictions, test_data[[dependent_variable]])
		svm_conf_matrices[[i]] <- svm_conf_matrix  # Store for debugging
#Check for NA Precision in SVM
		if (is.na(svm_conf_matrix$byClass['Precision'])) {
			cat("⚠️ Warning: NA Precision in SVM for iteration", i, "\n")
			print(svm_conf_matrix)
		}
		svm_metrics[i, ] <- c(
			svm_conf_matrix$overall['Accuracy'],
			    svm_conf_matrix$byClass['Precision'],
		    svm_conf_matrix$byClass['Sensitivity'],
		    svm_conf_matrix$byClass['Specificity'],
		    svm_conf_matrix$byClass['F1'],
		    auc(roc(test_data[[dependent_variable]], 
	            as.numeric(predict(svm_model, test_data, decision.values = TRUE))))	  )
# ---- Random Forest Model ----
		rf_model <- randomForest(as.formula(paste(dependent_variable, "~ .")), 
	                           data = train_data, ntree = 500, importance = TRUE)
		rf_predictions <- predict(rf_model, newdata = test_data)
		rf_predictions <- factor(rf_predictions, levels = levels(test_data[[dependent_variable]]))
	  
		rf_conf_matrix <- confusionMatrix(rf_predictions, test_data[[dependent_variable]])
		rf_conf_matrices[[i]] <- rf_conf_matrix  # Store for debugging
  # Check for NA Precision in Random Forest
		if (is.na(rf_conf_matrix$byClass['Precision'])) {
			cat("⚠️ Warning: NA Precision in Random Forest for iteration", i, "\n")
			print(rf_conf_matrix)
		}
		rf_metrics[i, ] <- c(
			rf_conf_matrix$overall['Accuracy'],
			rf_conf_matrix$byClass['Precision'],
			rf_conf_matrix$byClass['Sensitivity'],
			rf_conf_matrix$byClass['Specificity'],
			rf_conf_matrix$byClass['F1'],
			auc(roc(test_data[[dependent_variable]], 
	      	      predict(rf_model, newdata = test_data, type = "prob")[, 2])))
# ---- Decision Tree Model ----
		arbmax <- rpart(as.formula(paste(dependent_variable, "~ .")),
                data = train_data, method = "class",
                control = rpart.control(cp = 0.001, minsplit = 2, minsize = 1, xval = 5))
		source("functions.R")
		dt_model <- prune(arbmax, cp = cp.select(arbmax))
		dt_predictions <- predict(dt_model, newdata = test_data, type = "class")
		dt_predictions <- factor(dt_predictions, levels = levels(test_data[[dependent_variable]]))

		dt_conf_matrix <- confusionMatrix(dt_predictions, test_data[[dependent_variable]])
		dt_conf_matrices[[i]] <- dt_conf_matrix  # Store for debugging
#Check for NA Precision in Decision Tree
		if (is.na(dt_conf_matrix$byClass['Precision'])) {
			cat("⚠️ Warning: NA Precision in Decision Tree for iteration", i, "\n")
			print(nrow(dt_model$frame))  # Number of nodes in tree
			print(dt_conf_matrix)
		}
		dt_metrics[i, ] <- c(
			dt_conf_matrix$overall['Accuracy'],
			dt_conf_matrix$byClass['Precision'],
			dt_conf_matrix$byClass['Sensitivity'],
			dt_conf_matrix$byClass['Specificity'],
			dt_conf_matrix$byClass['F1'],
			auc(roc(test_data[[dependent_variable]], 
            	    predict(dt_model, newdata = test_data, type = "prob")[, 2])))

		selected_vars4 <- unique(c(dependent_variable, top_4_vars))
 		train_data_top4 <- train_data[, selected_vars4]
		test_data_top4 <- test_data[, selected_vars4]
		rf_model_top4 <- randomForest(as.formula(paste(dependent_variable, "~ .")),
                               data = train_data_top4,
                               ntree = 500,
                               importance = TRUE)
		rf_probabilities <- predict(rf_model_top4, newdata = test_data_top4, type = "prob")[, 2]
 
 		predictions <- predict(rf_model_top4, newdata = test_data_top4)
		conf_matrix <- confusionMatrix(predictions, test_data_top4[[dependent_variable]])
  		rf_top4_metrics$Accuracy[i] <- conf_matrix$overall['Accuracy']
		rf_top4_metrics$Precision[i] <- conf_matrix$byClass['Pos Pred Value']
		rf_top4_metrics$Sensitivity[i] <- conf_matrix$byClass['Sensitivity']
		rf_top4_metrics$Specificity[i] <- conf_matrix$byClass['Specificity']
		rf_top4_metrics$F1[i] <- conf_matrix$byClass['F1']
		rf_top4_metrics$AUC[i] <- auc(roc(test_data_top4[[dependent_variable]], rf_probabilities))

		selected_vars3 <- c(dependent_variable, top_3_vars)
		train_data_top3 <- train_data[, selected_vars3]
		test_data_top3 <- test_data[, selected_vars3]
  
		rf_model_top3 <- randomForest(as.formula(paste(dependent_variable, "~ .")),
                               data = train_data_top3,
                               ntree = 500,
                               importance = TRUE)
		rf_probabilities <- predict(rf_model_top3, newdata = test_data_top3, type = "prob")[, 2]
		predictions <- predict(rf_model_top3, newdata = test_data_top3)
		conf_matrix <- confusionMatrix(predictions, test_data_top3[[dependent_variable]])
  
		rf_top3_metrics$Accuracy[i] <- conf_matrix$overall['Accuracy']
		rf_top3_metrics$Precision[i] <- conf_matrix$byClass['Pos Pred Value']
		rf_top3_metrics$Sensitivity[i] <- conf_matrix$byClass['Sensitivity']
		rf_top3_metrics$Specificity[i] <- conf_matrix$byClass['Specificity']
		rf_top3_metrics$F1[i] <- conf_matrix$byClass['F1']
		rf_top3_metrics$AUC[i] <- auc(roc(test_data_top3[[dependent_variable]], rf_probabilities))
	} # End of iterations loop for models learning
	

	source("analyseResultats.R")


	
	
	
	
	





list.files(pattern = "\\.pdf$")

rf_model <- readRDS("rf_model.rds")
print(rf_model)

source("api.R")

colnames(rf_model$forest$xlevels)
colnames(rf_model$forest$votes)  # Lists features used in training

library(plumber)
pr <- plumb("api.R")  # Load the API definition
pr$run(host = "0.0.0.0", port = 8000)  # Run the API



