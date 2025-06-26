# prepare data
# Donnees thierry bege
# 14/04/2025


data_imputed <- read.csv("data_imputed.csv", na.strings = c("", "NA"),header=TRUE,fileEncoding = "latin1")

binary_columns <- c(
  "genre", "Type_accident","Chirurgie_Therapeutique", 
  "Tentative._suicide","patient_inconscient","Instabilite_hemodynamique",
  "leucocytes","tdm_pneumoperitoine","tdm_epanchement", "tdm_epanchement_hors_foie_rate",
  "tdm_epanchement_quantification","tdm_mesentere","tdm_mesent_infiltration","tdm_mesent_hematome",
  "tdm_mesent_extravasation","tdm_mesent_vascular_beeding","tdm_mesent_occlusion_vaisseaux", 
  "tdm_mesentere_ou_tube","tdm_tube","tdm_tube_hematome","tdm_tube_perforation","tdm_tube_paroi_epaissie","tdm_tube_paroi_amincie",
  "tdm_tube_.defaut_rehaussement","tdm_paroi_musculaire","tdm_Foie","tdm_rate","tdm_pancreas_duodenum","tdm_diaphragme","tdm_bassin",
  "tdm_urinaire","Score_FAGET_5","BIPS","BIPS._.2","ecchymose","vehicule", "position_vehicule", "ceinture", "airbag", "douleur", "defense")

# Convertir en facteur
	data_imputed[binary_columns] <- lapply(data_imputed[binary_columns], as.factor)
	data2 <- data_imputed[, !(names(data_imputed) %in% c("chirurgie", "X","missing_percentage","pec_initiale","Score._de_FAGET","BIPS._.2", "tdm_pneumoperitoine","tdm_epanchement", "tdm_epanchement_hors_foie_rate",
	                                                     "tdm_epanchement_quantification","tdm_mesentere","tdm_mesent_infiltration","tdm_mesent_hematome",
	                                                     "tdm_mesent_extravasation","tdm_mesent_vascular_beeding","tdm_mesent_occlusion_vaisseaux", 
	                                                     "tdm_mesentere_ou_tube","tdm_tube","tdm_tube_hematome","tdm_tube_perforation","tdm_tube_paroi_epaissie","tdm_tube_paroi_amincie",
	                                                     "tdm_tube_.defaut_rehaussement","tdm_paroi_musculaire","tdm_Foie","tdm_rate","tdm_pancreas_duodenum","tdm_diaphragme","tdm_bassin",
	                                                     "tdm_urinaire","Tentative._suicide",
	                                                     "defense",
	                                                     "Type_accident",
	                                                     "Instabilite_hemodynamique",
	                                                     "ceinture",
	                                                     "genre",
	                                                     "douleur",
	                                                     "vehicule"
	))]

str(data2)
#Vérifier le data frame après l'imputation
	head(data2)
# Verify that there are no more missing values
	print("missing data")
	print(any(is.na(data2)))


# analysing DT and Rf and computing variuables importances

#Set dependent and independent variables
	dependent_variable <- "Chirurgie_Therapeutique"
	independent_variables <- setdiff(colnames(data2), dependent_variable)

	arbmax <- rpart(as.formula(paste(dependent_variable, "~ .")),
             data = data2, method = "class",
                control = rpart.control(cp = 0.001, minsplit = 2, minsize = 1, xval = 5))
	dt_model <- prune(arbmax, cp = cp.select(arbmax))

	plot(dt_model,uniform=T,branch=0.5)
	text(dt_model,cex=0.8,use.n=T)

	rf_model <- randomForest(as.formula(paste(dependent_variable, "~ .")), 
	                           data = data2, ntree = 500, importance = TRUE)

#Extract importance values from the Random Forest model
	importance_values <- randomForest::importance(rf_model)
# Convert to a dataframe for easy manipulation
	importance_df <- as.data.frame(importance_values) %>%
	  tibble::rownames_to_column(var = "Variable")  # Add variable names as a column
# Order by MeanDecreaseAccuracy and select the top 20 variables
	top_9_vars <- importance_df %>%
	  arrange(desc(MeanDecreaseAccuracy)) %>%
	  head(9)


# Plot variable importance
	ggplot(top_9_vars, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
	  geom_bar(stat = "identity", fill = "steelblue") +
	  coord_flip() +  # Flip coordinates for better readability
	  labs(title = "Most Important Variables in Random Forest",
	       x = "Variables",
	       y = "Mean Decrease in Accuracy") +
	  theme_minimal()

# Order by MeanDecreaseAccuracy and select the top 4 variables
	top_4_vars <- importance_df %>%
	  arrange(desc(MeanDecreaseAccuracy)) %>%
	  head(4) %>%
	  pull(Variable)  # Extract variable names as a vector

	print(top_4_vars)

	top_3_vars <- importance_df %>%
	  arrange(desc(MeanDecreaseAccuracy)) %>%
	  head(3) %>%
	  pull(Variable)  # Extract variable names as a vector
	print(top_3_vars)



