library(readxl)
library(tidyverse)
library(textdata)
library(tidytext)
library(caret)

data <- read_excel(file.choose())

# Inspect the dataset
head(data)

# Add row numbers explicitly
data <- data %>%
  mutate(row_number = row_number())

# Tokenize comments
tokenized_data <- data %>%
  unnest_tokens(word, `USS Comment`)

# Load NRC lexicon
nrc_sentiments <- get_sentiments("nrc") %>%
  filter(sentiment %in% c("positive", "negative"))

# Join with NRC lexicon
sentiment_analysis <- tokenized_data %>%
  inner_join(nrc_sentiments, by = "word") %>%
  group_by(row_number) %>%
  summarise(predicted_sentiment = ifelse(sum(sentiment == "positive") > sum(sentiment == "negative"), "Positive", "Negative"))

# Merge predicted sentiment with actual sentiment
results <- data %>%
  left_join(sentiment_analysis, by = "row_number")

# Replace NAs in predicted_sentiment (if no sentiment words are found)
results <- results %>%
  mutate(predicted_sentiment = replace_na(predicted_sentiment, "Negative"))

# Evaluation metrics
confusion_matrix <- confusionMatrix(
  as.factor(results$predicted_sentiment),
  as.factor(results$Sentiment)
)

# Print evaluation metrics
print(confusion_matrix)


# Extract values from the confusion matrix
TP <- 40   # True Positives (Negative predicted as Negative)
FN <- 268  # False Negatives (Negative predicted as Positive)
FP <- 18   # False Positives (Positive predicted as Negative)
TN <- 340  # True Negatives (Positive predicted as Positive)

# Calculate Accuracy
accuracy <- (TP + TN) / (TP + TN + FP + FN)
print(paste("Accuracy:", round(accuracy, 4)))

# Calculate Precision (for Negative class, as it's the positive class in this context)
precision <- TP / (TP + FP)
print(paste("Precision:", round(precision, 4)))

# Calculate Recall (Sensitivity)
recall <- TP / (TP + FN)
print(paste("Recall (Sensitivity):", round(recall, 4)))

# Calculate F1-Score
f1_score <- 2 * ((precision * recall) / (precision + recall))
print(paste("F1-Score:", round(f1_score, 4)))
