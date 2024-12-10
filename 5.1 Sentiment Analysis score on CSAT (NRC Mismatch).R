library(readxl)
library(tidyverse)
library(textdata)
library(tidytext)

# Load the comments dataset
comments_data <- read_excel(file.choose())

# Filter comments: only English language and non-blank comments
comments_data <- comments_data %>%
  filter(Language == "english" & !is.na(`USS Comment`))

# Add a unique ID to the dataset
comments_data <- comments_data %>%
  mutate(id = row_number())

# Load the NRC lexicon
nrc_lexicon <- get_sentiments("nrc")

# Tokenize the comments
tokenized_data <- comments_data %>%
  unnest_tokens(word, `USS Comment`)

# Perform sentiment analysis using the NRC lexicon
nrc_sentiment_data <- tokenized_data %>%
  inner_join(nrc_lexicon, by = "word") %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

# Join NRC sentiment data back to the original dataset
comments_with_nrc_sentiment <- comments_data %>%
  left_join(nrc_sentiment_data, by = "id")

# Define mismatches based on sentiment score and satisfaction scale
mismatches <- comments_with_nrc_sentiment %>%
  filter(
    (`positive` > `negative` & `Average Response (calculated)` >= 4) |  # Positive sentiment with Dissatisfied or Very Dissatisfied
      (`negative` > `positive` & `Average Response (calculated)` <= 2)    # Negative sentiment with Satisfied or Very Satisfied
  )

# Count the number of mismatches
number_of_mismatches <- nrow(mismatches)

# Display the mismatches and the count
print(mismatches)
print(paste("Number of mismatches:", number_of_mismatches))
