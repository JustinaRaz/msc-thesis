# Linear mixed-effects
fit_models <- function(data, 
                       models, 
                       metrics, 
                       output_dir,
                       formula_spec,
                       RQ) {
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  summary_dir <- file.path(output_dir, "summaries")
  dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
  
  results_df <- data.frame(
    model = character(), 
    metric = character(), 
    term = character(), 
    estimate = numeric(),
    std_error = numeric(),
    t_value = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (model_name in models) {
    for (metric in metrics) {
      
      model_data <- data %>% filter(model == model_name)
      
      formula <- as.formula(paste(metric, "~", formula_spec))
      model <- lmer(formula, data = model_data)
      model_summary <- summary(model)
      
      coefficients <- model_summary$coefficients
      
      for (term in rownames(coefficients)) {
        results_df <- rbind(results_df, data.frame(
          model = model_name,
          metric = metric,
          term = term,
          estimate = coefficients[term, "Estimate"],
          std_error = coefficients[term, "Std. Error"],
          t_value = coefficients[term, "t value"],
          p_value = coefficients[term, "Pr(>|t|)"]
        ))
      }
      
      file_name <- paste0(RQ, model_name, "_", metric, "_summary.txt")
      
      file_path <- file.path(summary_dir, file_name)
      capture.output(model_summary, file = file_path)
    }
  }
  
  return(results_df)
}

# Bonferroni:
correct_pvalue <- function(data) {
  
  total_tests <- nrow(data)
  
  data %>%
    mutate(
      p_value_adjusted = pmin(p_value * total_tests, 1),
      p_value_adjusted = round(p_value_adjusted, 4),
      significance = case_when(
        p_value_adjusted < 0.001 ~ "***",
        p_value_adjusted < 0.01  ~ "**",
        p_value_adjusted < 0.05  ~ "*",
        TRUE ~ ""
      )
    )
}

# Save:
save_results <- function(data, output_dir, file_name) {
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  output_file <- file.path(output_dir, file_name)
  write_csv(data, output_file)
}