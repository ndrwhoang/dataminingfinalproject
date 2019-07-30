library(rpart) # Regression trees
library(rpart.plot) # Refression trees plot
library(gplots) # Correlation matrix plot
library(caret) 
library(tidyverse)
library(ranger) # Random Forest
library(dplyr)
library(rsample) # Data processing
library(xgboost) # Boosted tree
library(neuralnet) # Neural net
library(GGally)
set.seed(1)


# Import
setwd('D:/work/y3s2/datamining/final/insurance')
df <- read_csv('insurance.csv')

head(df)
summary(df)
dim(df)
t(t(names(df)))


# One-hot encoding
# Method for creating dummy vars
full_rank <- dummyVars(~., data=df, fullRank=TRUE)
# Applying method
df <- predict(full_rank, df)
# Converting back to dataframe
df <- as.data.frame(df)

# Conversion
df <- df %>% mutate_if(is.character, as.factor)
df$children <- as.numeric(df$children)

# Partition
train <- sample(nrow(df), nrow(df) * 0.8)
df.train <- df[train,]
df.valid <- df[-train,]

# correlation matrix with heatmap
colfun <- colorRampPalette(c("red", "white", "green"))
heatmap.2(round(cor(df.train[, c(1,3:4,7)]), 2), Rowv = FALSE, Colv = FALSE,
          dendrogram = "none", col = colfun(15), lwid=c(0.1,4), lhei=c(0.1,4),
          cellnote = round(cor(df.train[, c(1,3:4,7)]), 2), 
          notecol = "black", key = FALSE, trace = "none", margins = c(10, 10))


#BoxPlots
par(mfcol = c(3,1))
boxplot(df.train$charges ~ df.train$children, xlab = "Children", ylab = "Charges")
boxplot(df.train$charges ~ df.train$age, xlab = "Age", ylab = "Charges" )
boxplot(df.train$charges ~ df.train$smokeryes, xlab = "Smoker", ylab = "Charges" )


#************************************************************************************************************************************
#Regression Tree
df.rt <- rpart(charges ~ ., data = df.train)
df.rt
prp(df.rt, digits = 6, type = 1, extra = 1, varlen = -10,
    box.col = ifelse(df.rt$frame$var == "<leaf>", 'gray', 'white'))
#In-Sample Prediction
df.train.rt.pred <- predict(df.rt)
RMSE(df.train.rt.pred, df.train$charges)

#Out-of-Sample Prediction
df.valid.rt.pred <- predict(df.rt, newdata = df.valid)
data.frame(RMSE=RMSE(df.valid.rt.pred, df.valid$charges),
           MAE=MAE(df.valid.rt.pred, df.valid$charges),
           R2=R2(df.valid.rt.pred, df.valid$charges))

#************************************************************************************************************************************
## Random Forest
# Hyperparameter tuning
rf.grid <- expand.grid(mtry = seq(2, 4, by=1),
                       node_size = seq(2, 8, by=1),
                       sample_zise = c(0.5, 0.7),
                       OOB_RMSE = 0)

# Training
for(i in 1:nrow(rf.grid)) {
  df.rf <- ranger(charges~., data=df.train, num.trees=500,
                  mtry = rf.grid$mtry[i],
                  min.node.size = rf.grid$node_size[i],
                  sample.fraction = rf.grid$sample_zise[i])
  rf.grid$OOB_RMSE <- sqrt(df.rf$prediction.error)
}

# Prediction
df.rf.pred <- predict(df.rf, df.valid)

# Performance
data.frame(R2=R2(df.rf.pred$predictions, df.valid$charges), RMSE=RMSE(df.rf.pred$predictions, df.valid$charges),
           MAE=MAE(df.rf.pred$predictions, df.valid$charges))

#************************************************************************************************************************************

## Boosted machine

## This implementation of xgboost requires a matrix instead of dataframe
# Getting features
features <- setdiff(names(df.train), 'charges')
# Converting to matrix method
prep <- vtreat::designTreatmentsZ(df.train, features, verbose=FALSE)
new_vars <- prep %>%
  magrittr::use_series(scoreFrame) %>%
  dplyr::filter(code %in% c('clean', 'lev')) %>%
  magrittr::use_series(varName)
# Applying method on training set
df.train.features <- vtreat::prepare(prep, df.train, varRestriction=new_vars) %>% as.matrix()
df.train.response <- df.train$charges
# Applying method on validation set
df.valid.features <- vtreat::prepare(prep, df.valid, varRestriction=new_vars) %>% as.matrix()
df.valid.response <- df.valid$charges

# Hyperparameter tuning
grid.xgb <- expand.grid(eta=c(0.01, 0.03, 0.09),
                        max_depth=c(1,3,5),
                        min_child_weight=c(1,3,5),
                        subsample=c(0.5, 0.7),
                        colsample_bytree=0.8,
                        optimal_trees=0,
                        min_RMSE=0)


# XGBoost with tree as base learner with 10 fold cv 
for (i in 1:nrow(grid.xgb)) {
  params <- list(eta=grid.xgb$eta[i],
                 max_depth=grid.xgb$max_depth[i],
                 min_child_weight=grid.xgb$min_child_weight[i],
                 subsample=grid.xgb$subsample[i],
                 colsample_bytree=grid.xgb$colsample_bytree[i])
  
  df.xgb.cv <- xgb.cv(data=df.train.features, label=df.train.response, params=params, nrounds=1000,
                      nfold=10, objective='reg:linear', verbose=1, early_stopping_rounds=10)
  
  grid.xgb$optimal_trees[i] <- which.min(df.xgb.cv$evaluation_log$test_rmse_mean)
  grid.xgb$min_RMSE[i] <- min(df.xgb.cv$evaluation_log$test_rmse_mean)
}

# Performance table
grid.xgb %>% dplyr::arrange(min_RMSE) %>%
  head(10)

# Using hyperparameters with lowest RMSE
params <- list(eta=0.01,
               max_depth=3,
               min_child_weight=3,
               subsample=0.5,
               colsample_bytree=0.8)

# Training
df.xgb <- xgboost(params=params,
                  data=df.train.features,
                  label=df.train.response,
                  nrounds=564,
                  objective='reg:linear',
                  verbose=0)


# Predictions
df.xgb.pred <- predict(df.xgb, df.valid.features)
data.frame(R2=R2(df.xgb.pred, df.valid.response), RMSE=RMSE(df.xgb.pred, df.valid.response),
           MAE=MAE(df.xgb.pred, df.valid.response))

#******************************************************************************************************************************************

## Neural net



# Saving values to convert back after standardization
min <- as.data.frame(t(apply(df.train,2,min)))
max <- as.data.frame(t(apply(df.train,2,max)))

# Standardization
standardize <- preProcess(df.train, method='range')
df.train <- predict(standardize, df.train)
df.valid <- predict(standardize, df.valid)


# Model Building
# Building formula
formula <- reformulate(setdiff(colnames(df.train), 'charges'), response='charges')

# Fitting model
df.nn <- neuralnet(formula, data=df.train, hidden=c(3,3), rep=20, lifesign='minimal')

# Plotting
plot(df.nn, rep='best')

# Prediction
df.nn.pred <- compute(df.nn, df.valid[,1:8])$net.result

# Converting back to original values
df.nn.pred1 <- df.nn.pred*(max$charges-min$charges)+min$charges
df.valid$charges <- df.valid$charges*(max$charges-min$charges)+min$charges

# Result table
data.frame(R2=R2(df.nn.pred1, df.valid$charges), RMSE=RMSE(df.nn.pred1, df.valid$charges),
           MAE=MAE(df.nn.pred1, df.valid$charges))

#****************************************************************************************************************************************
#Combine Models
# Averaging 3 models
df.valid$predictions <- (df.rf.pred$predictions + df.nn.pred1 + df.xgb.pred)/3

# Getting performance
data.frame(RMSE=RMSE(df.valid$predictions, df.valid$charges), 
           MAE=MAE(df.valid$predictions, df.valid$charges), 
           R2=R2(df.valid$predictions, df.valid$charges))


