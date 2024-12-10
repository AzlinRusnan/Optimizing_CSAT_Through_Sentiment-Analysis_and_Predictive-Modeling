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
library(textclean)
library(textcat)

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

# Define a custom emoji dictionary
emoji_dict <- c(
  "😊" = "happy face",
  "👍" = "thumbs up",
  "😢" = "sad face",
  "😂" = "laughing face"
)

# Replace multiple spaces with a single space
df$`USS Comment` <- str_replace_all(df$`USS Comment`, "\\s+", " ")

# Trim leading and trailing spaces
df$`USS Comment` <- str_trim(df$`USS Comment`)

# Replace emojis with custom descriptions
df$`USS Comment` <- sapply(df$`USS Comment`, function(comment) {
  for (emoji in names(emoji_dict)) {
    comment <- gsub(emoji, emoji_dict[[emoji]], comment)
  }
  return(comment)
})

# Convert text to lowercase
df$`USS Comment` <- tolower(df$`USS Comment`)

# Remove non-alphabetic characters except spaces
df$`USS Comment` <- str_replace_all(df$`USS Comment`, "[^[:alpha:][:space:]]", "")

# Define stopwords
stopwords <- c("the", "and", "is", "in", "for", "on", "to", "a", "of", "it", "this", "would", "have", "very")

# Remove stopwords
df$`USS Comment` <- sapply(df$`USS Comment`, function(x) {
  words <- unlist(strsplit(x, " "))
  words <- words[!words %in% stopwords]
  paste(words, collapse = " ")
})

colSums(is.na(df))
head(df)


# Detect language of the 'USS Comment' column
df$Language <- textcat(df$`USS Comment`)

# Filter rows with English comments
df_english <- df[df$Language == "english", ]

tail(df)

write_xlsx(df, "Sentiment Analysis_cleaned_dataset.xlsx")
