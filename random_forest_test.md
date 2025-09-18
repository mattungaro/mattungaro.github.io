---
layout: default
---

# DECISION TREES #################
# PRACTICE 1 ##################

## KNN CHALLENGE DATA TO RANDOM FOREST #############


# INSTALL AND LOAD PACKAGES ################################

# Install pacman if you don't have it (uncomment next line)
# install.packages("pacman")

# Install and/or load packages with pacman
pacman::p_load(  # Use p_load function from pacman
  caret,         # Train/test functions
  e1071,         # Machine learning functions
  GGally,        # Plotting
  magrittr,      # Pipes
  mlbench,       # BreastCancer dataset
  pacman,        # Load/unload packages
  rio,           # Import/export data
  tidyverse,      # So many reasons
  rattle,           # pretty decision tree 
  randomForest,
  datasets
)
# Set random seed to reproduce the results
set.seed(5)

# LOAD AND PREPARE DATA ####################################

# Use the `spambase` datasets that were created previously 
# in "Spambase.R."
# Load data
data(BreastCancer)

# Summarize raw data
summary(BreastCancer)

# Prepare data
df <- BreastCancer %>%   # Save to `df`
  select(-Id) %>%        # Remove `Id` 
  rename(y = Class) %>%  # Rename `Class` to `y`
  mutate(                # Modify several variables
    across(              # Select several variables
      -y,                # Select all except `y`
      as.numeric         # Convert selected vars to numeric
    )
  ) %>%
  na.omit() %>%          # Omit cases with missing data
  as_tibble() %>%        # Save as tibble
  print()                # Show data in Console

# Split data into training (trn) and testing (tst) sets
df %<>% mutate(ID = row_number())  # Add row ID
trn <- df %>%                      # Create trn
  slice_sample(prop = .70)         # 70% in trn
tst <- df %>%                      # Create tst
  anti_join(trn, by = "ID") %>%    # Remaining data in tst
  select(-ID)                      # Remove id from tst
trn %<>% select(-ID)               # Remove id from trn
df %<>% select(-ID)                # Remove id from df

trn <- data.frame(trn)
# MODEL TRAINING DATA ######################################

# set training control parameters
ctrlparam <- trainControl(
  method  = "repeatedcv",   # method
  number  = 5,              # 5 fold
  repeats = 3               # 3 repeats
)

# Train decision tree on training data (takes a moment).
# First method tunes the complexity parameter.
rf <- randomForest( y ~ ., data=trn, 
                    proximity=TRUE) 
print(rf)


# Plot accuracy by complexity parameter values
rf %>% plot()
rf %>% plot(ylim = c(0, 1))  # Plot with 0-100% range



# Description of final training model
rf



# VALIDATE ON TEST DATA ####################################

# Predict on test set
pred <- rf %>%
  predict(newdata = tst)

# Accuracy of model on test data
cmtest <- pred %>%
  confusionMatrix(reference = tst$y)

# Plot the confusion matrix
cmtest$table %>% 
  fourfoldplot(color = c("red", "lightblue"), main = "Random Forest")

# Print the confusion matrix
cmtest %>% print()


# Confusion Matrix and Statistics
# 
# Reference
# Prediction  benign malignant
# benign       129         3
# malignant      3        70
# 
# Accuracy : 0.9707          
# 95% CI : (0.9374, 0.9892)

# My accuracy rate is way higher than 
# the previous tests of decision trees


MDSplot(rf, trn$y) 
MDSplot(rf, tst$y) 
