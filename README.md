# <div align="center"> Optimizing CSAT Through Sentiment Analysis and Predictive ML Techniques</div>

<div align="center"><img src="images/image.png" alt="Intro" /></div>

## 🚀 Project Overview

Customer Satisfaction (CSAT) is a crucial indicator of service quality in organizations. However, understanding the factors that contribute to customer dissatisfaction and predicting negative experiences remains a challenge. This project applies two advanced techniques to address this issue:

**🔹 Sentiment Analysis – Analyzing customer feedback to determine whether comments are positive or negative.**
   
**🔹 Predictive Modeling – Using machine learning to classify customer satisfaction levels based on incident-related data.**
   
By combining natural language processing (NLP) and machine learning, this study aims to provide deeper insights into customer feedback and proactively anticipate dissatisfaction. The goal is to enhance organizational response strategies and improve overall service efficiency.

## 🛠 Approach and Methodology

This project integrates sentiment analysis with predictive modeling to improve CSAT classification. The process includes:

📌 Sentiment Analysis – Textual feedback is analyzed and classified using:

✅ Lexicon-based methods (AFINN) for quick sentiment scoring.

✅ Deep learning (BERT) for contextual sentiment classification, capturing nuanced expressions.

📌 Feature Engineering – Incident-related variables such as Country, and Region are transformed into machine-readable formats.

📌 Machine Learning Models – The following classifiers were applied to predict CSAT:

✅ Logistic Regression (Traditional ML Baseline)

✅ Random Forest

✅ Support Vector Machine (SVM)

✅ Gradient Boosting Machine (GBM)

📌 Evaluation Metrics – Models were assessed using:

✅ Accuracy – Measures overall correctness of predictions.

✅ ROC-AUC Score – Evaluates the model's ability to distinguish between satisfied and dissatisfied customers.

✅ Precision, Recall, F1-score – Measures the balance between false positives and false negatives.

✅ Confusion Matrix – Provides insights into correct vs. misclassified instances.

🔍 Key Finding: Logistic Regression achieved the highest ROC-AUC (0.9512), demonstrating its superior ability to distinguish sentiment polarity, despite the high accuracy across all models (97.62%).


## 📈 Key Findings

✅ BERT significantly outperforms lexicon-based methods in understanding context, negations, and nuanced sentiment.

✅ Machine learning models achieved high accuracy (97.62%), proving their effectiveness in predicting customer satisfaction.

✅ Logistic Regression outperformed other models in ROC-AUC, making it the most effective model for sentiment classification.

## 🔮 Future Enhancements

🚀 Expand to Multilingual Feedback – Incorporate customer reviews in different languages to improve global applicability.

🚀 Feature Expansion – Add Ticket Priority, User Type, and additional metadata for better prediction accuracy.

🚀 Explore Advanced Transformers – Investigate more sophisticated NLP models for improved sentiment classification.
