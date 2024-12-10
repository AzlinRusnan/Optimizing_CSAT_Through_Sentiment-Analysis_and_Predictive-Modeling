library(readxl)
library(tidyverse)
library(textdata)
library(tidytext)

data <- read_excel(file.choose())
bing <- read_csv(file.choose())

# Convert lexicon to a data frame for matching
bing <- bing %>%
  mutate(sentiment = ifelse(sentiment == "positive", 1, -1))

# Function to calculate sentiment
calculate_sentiment <- function(text, lexicon) {
  words <- unlist(str_split(tolower(text), "\\s+"))
  sentiment_score <- sum(lexicon$sentiment[lexicon$word %in% words], na.rm = TRUE)
  if (sentiment_score > 0) {
    return("Positive")
  } else {
    return("Negative")
  }
}

# Apply sentiment analysis
data$Predicted_Sentiment <- sapply(data$`USS Comment`, calculate_sentiment, lexicon = bing)

levels(data$Sentiment)
levels(data$Predicted_Sentiment)

# Calculate metrics
library(caret)

# Convert factors for caret compatibility
data$Sentiment <- factor(data$Sentiment)
data$Predicted_Sentiment <- factor(data$Predicted_Sentiment, levels = levels(data$Sentiment))

data$Predicted_Sentiment <- factor(data$Predicted_Sentiment, levels = c("Positive", "Negative"))


confusion_matrix <- confusionMatrix(data$Predicted_Sentiment, data$Sentiment)

# Display metrics
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Pos Pred Value']
recall <- confusion_matrix$byClass['Sensitivity']
f1 <- 2 * (precision * recall) / (precision + recall)

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1))
print(confusion_matrix)

# Save results to a CSV file for review
write.csv(data, "Sentiment_Analysis_Results.csv")
