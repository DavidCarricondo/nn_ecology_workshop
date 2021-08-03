
### TOY EXAMPLE OF NEURAL NETWORK IN R  ###

# Loading the required packages

library(keras)
library(tfdatasets)
library(dplyr)
library(ggplot2)

## LOAD AND FORMAT THE DATA ####################################################
#Load dataset from the tfdatasets packages
boston_housing <- dataset_boston_housing()

#Split dataset in train and test data
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

paste0("Training entries: ", length(train_data), ", labels: ", 
       length(train_labels))

" Variables:
    *Per capita crime rate.
    *The proportion of residential land zoned for lots over 25,000 square feet.
    *The proportion of non-retail business acres per town.
    *Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    *Nitric oxides concentration (parts per 10 million).
    *The average number of rooms per dwelling.
    *The proportion of owner-occupied units built before 1940.
    *Weighted distances to five Boston employment centers.
    *Index of accessibility to radial highways.
    *Full-value property-tax rate per $10,000.
    *Pupil-teacher ratio by town.
    *1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
    *Percentage lower status of the population.
"

train_data[1, ]

train_labels[1:10] # Display first 10 entries


#Format the data
column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- train_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = train_labels)

test_df <- test_data %>% 
    as_tibble(.name_repair = "minimal") %>% 
    setNames(column_names) %>% 
    mutate(label = test_labels)



## PREPROCESSING ###############################################################

#Create the spec object
spec <- feature_spec(train_df, label ~ . ) %>% 
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
    fit()

spec

#Example of how to use a spec within an input layer for the NN
layer <- layer_dense_features(
    feature_columns = dense_features(spec), 
    dtype = tf$float32
)
layer(train_df)

### BUILD THE MODEL ############################################################

input <- layer_input_from_dataset(train_df %>% select(-label))

output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)

model %>% 
    compile(
        loss = "mse",
        optimizer = optimizer_rmsprop(),
        metrics = list("mean_absolute_error")
    )

## Same thing but putting everything into a function

build_model <- function() {
    input <- layer_input_from_dataset(train_df %>% select(-label))
    
    output <- input %>% 
        layer_dense_features(dense_features(spec)) %>% 
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 1) 
    
    model <- keras_model(input, output)
    
    model %>% 
        compile(
            loss = "mse",
            optimizer = optimizer_rmsprop(),
            metrics = list("mean_absolute_error")
        )
    
    model
}

### FIT THE MODEL   ############################################################

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat(".")
    }
)    

model <- build_model()

history <- model %>% fit(
    x = train_df %>% select(-label),
    y = train_df$label,
    epochs = 500,
    validation_split = 0.2,
    verbose = 0,
    callbacks = list(print_dot_callback)
)

plot(history)

## Little improvement after sometime, so let's try early stopping

# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- build_model()

history <- model %>% fit(
    x = train_df %>% select(-label),
    y = train_df$label,
    epochs = 500,
    validation_split = 0.2,
    verbose = 0,
    callbacks = list(early_stop)
)

plot(history)

c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, 
                                      verbose = 0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))


### PREDICTIONS   #############################################################
test_predictions <- model %>% predict(test_df %>% select(-label))
test_predictions[ , 1]
(test_df %>% select(label))-test_predictions


