# Nombre de répétitions de l’optimisation
n_optim_runs <- 30

# Initialisation dataframe pour stocker les résultats
optim_results <- data.frame(
  run = 1:n_optim_runs,
  nodesize = numeric(n_optim_runs),
  Accuracy = numeric(n_optim_runs),
  Precision = numeric(n_optim_runs),
  Recall = numeric(n_optim_runs),
  Specificity = numeric(n_optim_runs),
  F1 = numeric(n_optim_runs),
  AUC = numeric(n_optim_runs)
)

for (j in 1:n_optim_runs) {
  set.seed(j * 99)  # Pour varier un peu
  
  # 1. Paramètre à optimiser
  par_set_nodesize <- makeParamSet(
    makeIntegerParam("nodesize", lower = 50, upper = 100)
  )
  
  # 2. Fonction objectif
  obj_fun_nodesize <- function(x) {
    rf <- randomForest(
      as.formula(paste(dependent_variable, "~ .")),
      data = train_data_top3,
      nodesize = x$nodesize,
      ntree = 500,
      importance = TRUE
    )
    pred <- predict(rf, newdata = test_data_top3)
    conf <- confusionMatrix(pred, test_data_top3[[dependent_variable]])
    return(conf$byClass["Sensitivity"])  # <- ici on maximise le Recall
  }
  
  # 3. Fonction objectif wrap
  obj_fun_wrapped_nodesize <- makeSingleObjectiveFunction(
    name = "rf_recall",
    fn = obj_fun_nodesize,
    par.set = par_set_nodesize,
    has.simple.signature = FALSE,
    minimize = FALSE
  )
  
  # 4. Design initial
  initial_design_nodesize <- generateDesign(
    n = 20,
    par.set = par_set_nodesize,
    fun = lhs::randomLHS
  )
  initial_design_nodesize$y <- apply(initial_design_nodesize, 1, function(params) {
    obj_fun_nodesize(as.list(params))
  })
  
  # 5. MBO control
  ctrl <- makeMBOControl()
  ctrl <- setMBOControlTermination(ctrl, iters = 50)
  
  # 6. Optimisation
  res_nodesize <- mbo(
    fun = obj_fun_wrapped_nodesize,
    design = initial_design_nodesize,
    control = ctrl,
    show.info = FALSE
  )
  
  # 7. Meilleur modèle avec paramètre optimisé
  best_nodesize <- res_nodesize$x$nodesize
  
  final_rf <- randomForest(
    as.formula(paste(dependent_variable, "~ .")),
    data = train_data_top3,
    nodesize = best_nodesize,
    ntree = 500,
    importance = TRUE
  )
  
  final_preds <- predict(final_rf, newdata = test_data_top3)
  final_probs <- predict(final_rf, newdata = test_data_top3, type = "prob")[, 2]
  conf_final <- confusionMatrix(final_preds, test_data_top3[[dependent_variable]])
  
  # 8. Enregistrer les résultats
  optim_results[j, ] <- c(
    j,
    best_nodesize,
    conf_final$overall["Accuracy"],
    conf_final$byClass["Pos Pred Value"],
    conf_final$byClass["Sensitivity"],
    conf_final$byClass["Specificity"],
    conf_final$byClass["F1"],
    auc(roc(test_data_top3[[dependent_variable]], final_probs))
  )
}



summary(optim_results)

# Moyenne et écart-type des métriques
sapply(optim_results[, 3:8], function(x) c(mean = mean(x), sd = sd(x)))



# Boxplot des performances
optim_results_long <- reshape2::melt(optim_results[, -1], id.vars = "nodesize")
ggplot(optim_results_long, aes(x = variable, y = value)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Variabilité des performances (10 runs)",
       x = "Métrique", y = "Valeur") +
  theme_minimal()

# Histogramme des nodesize choisis
ggplot(optim_results, aes(x = nodesize)) +
  geom_histogram(binwidth = 2, fill = "coral", color = "black") +
  labs(title = "Distribution des nodesize optimisés",
       x = "nodesize", y = "Fréquence")


