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
library(tidyr)
library(writexl)
library(hunspell)


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

# Step 1: Replace multiple spaces with a single space
df$"USS Comment" <- str_replace_all(df$"USS Comment", "\\s+", " ")

# Step 1.1: Replace slash ("/") with a space to avoid words being concatenated
df$"USS Comment" <- str_replace_all(df$"USS Comment", "/", " ")

# Step 2: Trim leading and trailing spaces
df$"USS Comment" <- str_trim(df$"USS Comment")

# Step 3: Remove email addresses using a regular expression
df$"USS Comment" <- str_replace_all(df$"USS Comment", "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "")

# Step 4: Remove URLs using a regular expression
df$"USS Comment" <- str_replace_all(df$"USS Comment", "https?://\\S+|www\\S+", "")

# Step 5: Remove @ and numbers, but keep ' , !, ?
df$"USS Comment" <- str_replace_all(df$"USS Comment", "[@\\d]", "")

# Step 6: Remove common sign-off phrases (e.g., "Warm regards", "Best regards", "Kind regards")
df$"USS Comment" <- str_replace_all(df$"USS Comment", "(Warm regards|Best regards|Kind regards|Sincerely|Cheers|Best wishes)\\s*,?\\s*", "")

# Step 7: Remove unwanted patterns like "[cid:image.png" or any similar pattern
df$"USS Comment" <- str_replace_all(df$"USS Comment", "\\[cid:image\\.png[^\\]]*\\]", "")

# Step 8: Remove unwanted characters like -, %, :, *, .
df$"USS Comment" <- str_replace_all(df$"USS Comment", "[-%:*.,\\[\\]/\"()]", "")

# Step 9: Replace multiple exclamation marks with a single one
df$"USS Comment" <- str_replace_all(df$"USS Comment", "!+", "!")

# Step 10: Remove comments that contain only punctuation marks (e.g., "!!!", "????")
df$"USS Comment" <- ifelse(str_detect(df$"USS Comment", "^[!?]+$"), "", df$"USS Comment")

emoji_dict <- c(
  "😊" = "happy face",
  "👍" = "thumbs up",
  "😢" = "sad face",
  "😂" = "laughing face",
  "☹" = "frowning face",
  "🙂" = "smiling face"
)


# Replace emojis with custom descriptions
df$`USS Comment` <- sapply(df$`USS Comment`, function(comment) {
  for (emoji in names(emoji_dict)) {
    comment <- gsub(emoji, emoji_dict[[emoji]], comment)
  }
  return(comment)
})

# --- Spell checking and correction --- #

# Function to correct spelling in a single comment
correct_spelling <- function(comment) {
  if (is.na(comment) || comment == "") {
    return(comment) # Skip if the comment is empty or NA
  }
  
  # Split the comment into words
  words <- unlist(strsplit(comment, "\\s+"))
  
  # Check for misspelled words
  misspelled <- hunspell(words)
  
  # Replace misspelled words with the first suggestion
  for (i in seq_along(misspelled)) {
    if (length(misspelled[[i]]) > 0) {
      suggestions <- hunspell_suggest(misspelled[[i]])
      if (length(suggestions[[1]]) > 0) {
        words[words == misspelled[[i]]] <- suggestions[[1]][1]
      }
    }
  }
  
  # Recombine the words into a corrected comment
  corrected_comment <- paste(words, collapse = " ")
  return(corrected_comment)
}

# Apply spell correction to the "USS Comment" column
df$"Corrected USS Comment" <- sapply(df$"USS Comment", correct_spelling)


# View the cleaned comments
print(df)

write_xlsx(df, "Sentiment Analysis_cleaned_dataset_4.1_Sept.xlsx")

##################################################################

library(textcat)

# Detect language of the 'USS Comment' column
df$Language <- textcat(df$`USS Comment`)

# Filter rows with English comments
df_english <- df[df$Language == "english", ]

tail(df)

write_xlsx(df, "english_cleaned_dataset.xlsx")


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

# Filter the mismatched comments
mismatched_comments <- df %>% filter(sentiment_check == "Mismatch")

# View the mismatched comments
tail(mismatched_comments)


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
