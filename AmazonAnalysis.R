library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)


trainData <- vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("test.csv")


#ggplot(data = baked, aes(x = ROLE_ROLLUP_1, y = ACTION)) +
#  geom_boxplot()



my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
#  step_mutate(ACTION = as.factor(ACTION)) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#  step_mutate(across(all_numeric_predictors(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
#  step_dummy(all_nominal_predictors()) #%>% # dummy variable encoding
#  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding (must be 2-f
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding (must be 2-f

# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)



# type of model
logRegModel <- logistic_reg() %>%
  set_engine("glm")

#put into a workflow here

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = trainData) #!!!!! ERROR

# make predictions
amazon_predictions <- predict(logReg_workflow,
                              new_data = testData,
                              type = "prob")
## with type="prob" amazon_predictions will have 2 columns12
## one for Pr(0) and the other for Pr(1)!13
## with type="class" it will just have one column (0 or 1)


kaggle.amazon.preds <- amazon_predictions %>%
  bind_cols(., testData) %>%
  select(id, .pred_1) %>%
  rename(Id = id) %>%
  rename(Action = .pred_1)

vroom_write(x=kaggle.amazon.preds, file="./kaggle_amazon_preds.csv", delim=",")




#### Penalized Logistic Regression
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)

# import data
trainData <- vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("test.csv")

#recipe
my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding (must be 2-f
  step_normalize(all_numeric_predictors())

#model
pen.log.reg <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

#workflow
amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pen.log.reg)

## Grid of values to tune over

L = 4
K = 5

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = L) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(trainData, v = K, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics = metric_set(roc_auc))

#          metrics=metric_set(roc_auc, f_meas, sens, recall, yardstick::spec,
#                             precision, accuracy)) #Or leave metrics NULL


# Fine Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow & fit it
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# Predict
mypreds.tune <- predict(final_wf, new_data = testData)

# For submission
kaggle.amazon.plr.preds <- mypreds.tune %>%
  bind_cols(., testData) %>%
  select(id, .pred_1) %>%
  rename(Id = id) %>%
  rename(Action = .pred_1)

vroom_write(x=kaggle.amazon.plr.preds, file="./kaggle_amazon_plr_preds.csv", delim=",")


#### Regression Tree - RFsBinary

library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)

# import data
trainData <- vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("test.csv")

#recipe
my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding (must be 2-f
  step_normalize(all_numeric_predictors())

#model
amazon.rf.mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#workflow
amazon_rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(amazon.rf.mod)


# Grid of values to tune over
L = 5 # number of penalties and mixure
K = 5 # number of folds
grid_of_tuning_params <- grid_regular(mtry(range = c(1, 30)),
                                      min_n(),
                                      levels = L)

folds <- vfold_cv(trainData, v = K, repeats = 1)

# Run the CV
CV_results <- amazon_rf_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(roc_auc))


# Fine Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow & fit it
final_wf <- amazon_rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# Predict
mypreds.tune <- predict(final_wf, new_data = testData)

# For submission
kaggle.amazon.rf.preds <- mypreds.tune %>%
  bind_cols(., testData) %>%
  select(id, .pred_class) %>%
  rename(Id = id) %>%
  rename(Action = .pred_class)

vroom_write(x=kaggle.amazon.rf.preds, file="./kaggle_amazon_rf_preds.csv", delim=",")




#### KNN Model
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)

# import data
trainData <- vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("test.csv")

#recipe
my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding (must be 2-f
  step_normalize(all_numeric_predictors())


#model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


#workflow
knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)


# Grid of values to tune over
L = 5 # number of penalties and mixure
K = 5 # number of folds
grid_of_tuning_params <- grid_regular(neighbors(),
                                      levels = L)

folds <- vfold_cv(trainData, v = K, repeats = 1)

# Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(roc_auc))


# Fine Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# predict
my.knn.preds <- predict(final_wf, new_data = testData, type = 'prob')


# For submission
kaggle.amazon.knn.preds <- my.knn.preds %>%
  bind_cols(., testData) %>%
  select(id, .pred_1) %>%
  rename(Id = id) %>%
  rename(Action = .pred_1)

vroom_write(x=kaggle.amazon.knn.preds, file="./kaggle_amazon_knn_preds.csv", delim=",")



#### Naive Bayes
library(discrim)
library(naivebayes)


##nb model
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)


# Grid of values to tune over
L = 5 # number of penalties and mixure
K = 5 # number of folds
grid_of_tuning_params <- grid_regular(Laplace(),
                                      smoothness(),
                                      levels = L)

folds <- vfold_cv(trainData, v = K, repeats = 1)

# Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(roc_auc))


# Fine Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# predict
my.nb.preds <- predict(final_wf, new_data = testData, type = 'prob')

# For submission
kaggle.amazon.nb.preds <- my.nb.preds %>%
  bind_cols(., testData) %>%
  select(id, .pred_1) %>%
  rename(Id = id) %>%
  rename(Action = .pred_1)

vroom_write(x=kaggle.amazon.nb.preds, file="./kaggle_amazon_nb_preds.csv", delim=",")


