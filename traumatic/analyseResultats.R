# compute metrics and produce latex tables and roc curves


# Calculate average metrics
	avg_svm_metrics <- colMeans(svm_metrics, na.rm = TRUE)
	avg_rf_metrics <- colMeans(rf_metrics, na.rm = TRUE)
	avg_dt_metrics <- colMeans(dt_metrics, na.rm = TRUE)
	avg_rf_top3_metrics <- colMeans(rf_top3_metrics, na.rm = TRUE)

# Calculer les écarts types
	sd_svm_metrics <- apply(svm_metrics, 2, sd,na.rm= TRUE )
	sd_rf_metrics <- apply(rf_metrics, 2, sd, na.rm= TRUE)
	sd_dt_metrics <- apply(dt_metrics, 2, sd,na.rm= TRUE)
	sd_rf_top3_metrics <- apply(rf_top3_metrics, 2, sd,na.rm= TRUE) 

# Combiner les résultats dans un tableau récapitulatif
	results_table <- data.frame(
		Metric = c("Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC"),
	  	SVM_Mean = avg_svm_metrics,
		SVM_SD = sd_svm_metrics,
		RandomForest_Mean = avg_rf_metrics,
		RandomForest_SD = sd_rf_metrics,
		DecisionTree_Mean = avg_dt_metrics,
		DecisionTree_SD = sd_dt_metrics,
		RandomForest_top3_MEAN = avg_rf_top3_metrics,
		RandomForest_top3_SD = sd_rf_top3_metrics)

# Convert metrics into a single data frame
	metrics_df <- data.frame(
	  Iteration = rep(1:n_iterations, each = 6 * 4),  # 6 metrics × 4 models
	  Model = rep(rep(c("SVM", "Random Forest", "Decision Tree", "RandomForest_top4"), each = 6), times = n_iterations),
	  Metric = rep(c("Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC"), times = 4 * n_iterations),
	  Value = c(as.vector(t(svm_metrics)), as.vector(t(rf_metrics)), as.vector(t(dt_metrics)), as.vector(t(rf_top4_metrics)))  # Ensure correct alignment
	)

# Ensure `Model` and `Metric` are factors for proper ordering in plots
	metrics_df$Model <- factor(metrics_df$Model, levels = c("SVM", "Random Forest", "Decision Tree", "RandomForest_top4"))
	metrics_df$Metric <- factor(metrics_df$Metric, levels = c("Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC"))





# Remove NA values to avoid ggplot warnings
	metrics_df <- na.omit(metrics_df)


# Plot the Boxplots for Each Metric Across Models
	ggplot(metrics_df, aes(x = Model, y = Value, fill = Model)) +
		geom_boxplot() +
		facet_wrap(~Metric, scales = "free_y") +  # Separate plots for each metric
		labs(title = "Model Performance Comparison",
		x = "Model",
		y = "Metric Value") +
		theme_minimal() +
		theme(legend.position = "none",
		axis.text.x = element_text(angle = 45, hjust = 1))

# Arrondir les valeurs à 4 décimales
	results_table[, 2:5] <- round(results_table[, 2:5], 4)
# Afficher le tableau

# ---- Arrondir les résultats et afficher le tableau ----
	results_table[, -1] <- round(results_table[, -1], 4)
	print(results_table)

# Optionnel : Créer une table LaTeX

	latex_table <- xtable(results_table, 
                      caption = "Comparaison des métriques moyennes pour les différents modèles",
                      label = "tab:model_comparison")
# Imprimer la table LaTeX
#	print(latex_table, type = "latex", include.rownames = FALSE, caption.placement = "top", table.placement = "htb")


################### Courbes ROC 
# Créer une liste pour stocker les objets ROC
	roc_list <- list()
# ---- SVM Model ----
	svm_probabilities <- predict(svm_model, newdata = test_data, probability = TRUE)
	svm_probabilities <- attr(svm_probabilities, "probabilities")[, 2]  # Probabilité pour la classe positive
	roc_list$SVM <- roc(response = test_data[[dependent_variable]], predictor = svm_probabilities)

# ---- Random Forest Model ----
	rf_probabilities <- predict(rf_model, newdata = test_data, type = "prob")[, 2]  # Probabilité pour classe 1
	roc_list$RandomForest <- roc(as.numeric(test_data[[dependent_variable]]), rf_probabilities)

# ---- Decision Tree Model ----
	dt_probabilities <- predict(dt_model, newdata = test_data, type = "prob")[, 2]
	roc_list$DecisionTree <- roc(as.numeric(test_data[[dependent_variable]]), dt_probabilities)

# ---- Random Forest (Top 4 variables) ----
	rf_top4_probabilities <- predict(rf_model_top4, newdata = test_data_top4, type = "prob")[, 2]
	roc_list$RandomForest_top4 <- roc(as.numeric(test_data_top4[[dependent_variable]]), rf_top4_probabilities)

# ---- Random Forest (Top 3 variables) ----
	rf_top3_probabilities <- predict(rf_model_top3, newdata = test_data_top3, type = "prob")[, 2]
	roc_list$RandomForest_top3 <- roc(as.numeric(test_data_top3[[dependent_variable]]), rf_top3_probabilities)

# Créer un dataframe pour ggplot
	roc_data <- data.frame()
	for (model in names(roc_list)) {
		roc_curve <- roc_list[[model]]
		roc_data <- rbind(roc_data, data.frame(
		    Model = model,
		    FPR = 1 - roc_curve$specificities,
		    TPR = roc_curve$sensitivities  ))
	}


# ---- Tracer les courbes ROC ----
	print(ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
	  geom_line(linewidth = 1) +
	  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50", alpha = 0.7) +
	  labs(title = "ROC Curves for Different Models",
	       x = "False Positive Rate (1 - Specificity)",
	       y = "True Positive Rate (Sensitivity)",
	       color = "Model") +
	  scale_color_brewer(palette = "Set1") +
	  coord_equal() +
	  theme_minimal() +
	  theme(
	    legend.position = "bottom",
	    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
	    axis.title = element_text(face = "bold", size = 12),
	    axis.text = element_text(size = 10),
	    legend.title = element_text(face = "bold"),
	    panel.grid.minor = element_blank(),
	    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5)
	  ) +
	  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
	  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0, 0))
)

