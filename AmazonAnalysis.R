library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(ggmosaic)

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
#  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
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

L = 5
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
          metrics = NULL)

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


