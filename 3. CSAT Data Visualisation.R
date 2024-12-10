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
library(plotly)

# Read the data 
df <- read_excel(file.choose())

head(df)
colnames(df)

##########################################
##   Blank vs. Non-Blank Over Time      ##
##########################################

# Add a column to classify comments as Blank or Non-Blank
df$Comment_Status <- ifelse(is.na(df$`USS Comment`) | df$`USS Comment` == "", "Blank", "Non-Blank")

# Group by Month-Year and Comment_Status
status_summary <- df %>%
  group_by(`Year`, Comment_Status) %>%
  summarise(Count = n(), .groups = 'drop')

# Check the summarized data
head(status_summary)

# Create the ggplot object
plot <- ggplot(status_summary, aes(x = `Year`, y = Count, fill = Comment_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(
    title = "Blank vs. Non-Blank Comments Over Time",
    x = "Year",
    y = "Count",
    fill = "Comment Status"
  ) +
  scale_fill_manual(
    values = c("Blank" = "#FF6666", "Non-Blank" = "#66B2FF") # Custom colors
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10)
  )

# Convert the ggplot to an interactive plotly plot
interactive_plot <- ggplotly(plot)

# Display the plot
interactive_plot

##########################################
##     Languages in the Dataset         ##
##########################################

# Filter out blank comments and prepare hierarchical data
filtered_data <- df %>%
  filter(trimws(`USS Comment`) != "" & !is.na(`USS Comment`)) %>% # Exclude blanks
  group_by(Language, Country) %>% # Assuming there's a "Country" column
  summarise(Count = n(), .groups = 'drop')

# Prepare hierarchical data for sunburst
language_counts <- filtered_data %>%
  group_by(Language) %>%
  summarise(Count = sum(Count), .groups = 'drop')

# Combine Language and Country data
sunburst_data <- bind_rows(
  data.frame(labels = language_counts$Language, parents = "", values = language_counts$Count),
  data.frame(labels = filtered_data$Country, parents = filtered_data$Language, values = filtered_data$Count)
)

# Create the sunburst plot
sunburst_plot <- plot_ly(
  data = sunburst_data,
  type = 'sunburst',
  labels = ~labels,
  parents = ~parents,
  values = ~values,
  branchvalues = 'total'
) %>%
  layout(
    title = "Language and Country Distribution"
  )

# Display the plot
sunburst_plot

##########################################
##            Text Mining               ##
##########################################

# Create a corpus
corpus <- Corpus(VectorSource(df$`USS Comment`))

# Replacing "/", "@" and "|" with space
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "\\|")
corpus <- tm_map(corpus, content_transformer(tolower)) 
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, c("just","like","will","can", "thank")) 
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)

inspect(corpus[[2]])

# Create a TermDocumentMatrix
tdm <- TermDocumentMatrix(corpus)

# Calculate the term frequency
term_freq <- rowSums(as.matrix(tdm))
term_freq <- sort(term_freq, decreasing = TRUE)
head(term_freq, 5)

# Create a data frame of terms and their frequencies
df <- data.frame(word = names(term_freq), freq = term_freq)

##########################################
##        Most Frequent Words           ##
##########################################

top_words <- df %>% 
  arrange(desc(freq)) %>% 
  head(n = 5)

ggplot(top_words, aes(x=reorder(word, -freq), y=freq)) +
  geom_bar(stat="identity", fill="skyblue", colour="black") +
  geom_text(aes(label=freq), vjust=-0.3, size=4) + # Add frequency labels on top of bars
  labs(y="Frequency", title="Top 5 Words by Frequency") +
  theme_minimal() + # Use a minimal theme for a cleaner look
  theme(
    axis.text.x=element_text(angle=45, hjust=1, size=12, face="bold"), # Style x-axis text
    axis.title.x=element_blank(),
    axis.title.y=element_text(size=14, face="bold"), # Style y-axis title
    plot.title=element_text(hjust=0.5, size=16, face="bold") # Center and style plot title
  )

##########################################
##            Word Cloud                ##
##########################################

# Generate the word cloud
set.seed(1234)
top_n <- head(df, 200)  
wordcloud2(
  data = top_n,
  size = 1.6,
  minSize = 0.9,
  color = 'random-light',
  backgroundColor = "black",
  shape = "diamond",
  fontFamily = "HersheySymbol",
  rotateRatio = 0.3, # Add some rotation to the words
  ellipticity = 0.65 # Adjust the shape for better fit
)


