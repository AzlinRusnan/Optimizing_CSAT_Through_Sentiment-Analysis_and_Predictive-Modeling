library(readxl)
library(tidyverse)
library(textdata)
library(tidytext)

comments_data <- read_excel(file.choose())
bing_lexicon <- read_csv(file.choose())

# Filter comments: only English language and non-blank comments
comments_data <- comments_data %>%
  filter(Language == "english" & !is.na(`USS Comment`))

# Add a unique ID to the dataset
comments_data <- comments_data %>%
  mutate(id = row_number())

# Remove duplicates in the Bing lexicon
bing_lexicon <- bing_lexicon %>%
  distinct(word, .keep_all = TRUE)

# Tokenize the comments and perform sentiment analysis
tokenized_data <- comments_data %>%
  unnest_tokens(word, `USS Comment`)

# Perform sentiment analysis using the Bing lexicon
sentiment_data <- tokenized_data %>%
  inner_join(bing_lexicon, by = "word") %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative)

# Join sentiment data back to the original dataset
comments_with_sentiment <- comments_data %>%
  left_join(sentiment_data, by = "id")

# Filter mismatches based on the clarified scale
mismatches <- comments_with_sentiment %>%
  filter(
    (sentiment_score > 0 & `Average Response (calculated)` >= 4) |  # Positive sentiment with Dissatisfied or Very Dissatisfied
      (sentiment_score < 0 & `Average Response (calculated)` <= 2)    # Negative sentiment with Satisfied or Very Satisfied
  )

# Count the number of mismatches
number_of_mismatches <- nrow(mismatches)

# Display the mismatches and the count
print(mismatches)
print(paste("Number of mismatches:", number_of_mismatches))
