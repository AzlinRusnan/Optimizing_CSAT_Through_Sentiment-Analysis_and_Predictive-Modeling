library(readxl)
library(tidyverse)
library(textdata)
library(tidytext)
library(caret)

data <- read_excel(file.choose())
head(data)

# Add row numbers explicitly
data <- data %>%
  mutate(row_number = row_number())

# Tokenize comments
tokenized_data <- data %>%
  unnest_tokens(word, `USS Comment`)

# Load AFINN lexicon
afinn_sentiments <- get_sentiments("afinn")

# Join with AFINN lexicon
sentiment_analysis <- tokenized_data %>%
  inner_join(afinn_sentiments, by = "word") %>%
  group_by(row_number) %>%
  summarise(sentiment_score = sum(value)) %>%
  mutate(predicted_sentiment_afinn = ifelse(sentiment_score > 0, "Positive", "Negative"))

# Merge predicted sentiment with actual sentiment
results <- data %>%
  left_join(sentiment_analysis, by = "row_number")

# Evaluation metrics
confusion_matrix <- confusionMatrix(
  as.factor(results$predicted_sentiment_afinn),
  as.factor(results$Sentiment)
)

# Print confusion matrix and metrics
print(confusion_matrix)

# Extract values from the confusion matrix
TP <- 563   # True Positives (Negative predicted as Negative)
FN <- 36  # False Negatives (Negative predicted as Positive)
FP <- 21   # False Positives (Positive predicted as Negative)
TN <- 14  # True Negatives (Positive predicted as Positive)

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
