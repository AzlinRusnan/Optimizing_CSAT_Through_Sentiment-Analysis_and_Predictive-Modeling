library(readxl)
library(stringr)
library(NLP)
library(tm)
library(SnowballC)
library(RColorBrewer)
library(wordcloud)
library(syuzhet)
library(ggplot2)
library(wordcloud2)
library(lubridate)
library(dplyr)
library(tidytext)
library(reshape2)
library(tidyr) # Load tidyr for unnest_tokens

# Read the data 
df <- read_excel(file.choose())
# View the first few rows
head(df)
# Structure
str(df)

# View the column names to ensure accuracy
colnames(df)

# check missing values
colSums(is.na(df))

# --- Data cleaning --- #
# Remove extra spaces and newline characters
df$`USS Comment` <- str_replace_all(df$`USS Comment`, "\\s+", " ")  # Replace multiple spaces with a single space
df$`USS Comment` <- str_trim(df$`USS Comment`)  # Trim leading and trailing spaces

# Remove rows where "USS Comment" is NA or empty
df <- df %>% filter(!is.na(`USS Comment`) & `USS Comment` != "")

# Convert the comments to lowercase
df$`USS Comment` <- tolower(df$`USS Comment`)

# Remove special characters and numbers (optional)
df$`USS Comment` <- str_replace_all(df$`USS Comment`, "[^[:alpha:][:space:]]", "")

# If you have a stopwords list, remove those words
stopwords <- c("the", "and", "is", "in", "for", "on", "to", "a", "of", "it", "this")  
df$`USS Comment` <- sapply(df$`USS Comment`, function(x) {
  words <- unlist(strsplit(x, " "))
  words <- words[!words %in% stopwords]
  paste(words, collapse = " ")
})

# Sentiment Analysis #

# Apply sentiment analysis to the USS Comment column
df$sentiment <- get_sentiment(df$`USS Comment`, method = "syuzhet")

# View the first few rows with sentiment analysis
head(df)

# Categorize sentiment based on sentiment score
df$sentiment_category <- case_when(
  df$sentiment > 0 ~ "Positive",
  df$sentiment == 0 ~ "Neutral",
  df$sentiment < 0 ~ "Negative"
)

# Create matching logic for sentiment and Average Response
df$sentiment_check <- case_when(
  df$sentiment_category == "Positive" & df$`Average Response (calculated)` %in% c(1, 2) ~ "Match",
  df$sentiment_category == "Negative" & df$`Average Response (calculated)` %in% c(4, 5) ~ "Match",
  df$sentiment_category == "Neutral" & df$`Average Response (calculated)` == 3 ~ "Match",
  TRUE ~ "Mismatch"
)

# View the mismatch summary
table(df$sentiment_check)

# Visualize sentiment and rating mismatches
ggplot(df, aes(x = sentiment_check, fill = sentiment_check)) +
  geom_bar(stat = "count", show.legend = FALSE, width = 0.6) +  # Adjust bar width
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.3, size = 5) +  # Add count labels
  scale_fill_manual(values = c("Match" = "#4CAF50", "Mismatch" = "#FF5722")) +  # Custom colors
  labs(
    title = "Sentiment and Rating Mismatches",
    subtitle = "Comparison between comment sentiment and rating given",
    x = "Mismatch Status",
    y = "Count"
  ) +
  theme_minimal() +  # Clean background
  theme(
    text = element_text(size = 14),  # Adjust text size
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Title formatting
    plot.subtitle = element_text(hjust = 0.5, size = 12),  # Subtitle formatting
    axis.title = element_text(size = 14),  # Axis title size
    axis.text = element_text(size = 12)  # Axis text size
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for clarity
