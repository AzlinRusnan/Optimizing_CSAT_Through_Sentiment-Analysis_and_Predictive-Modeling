# <div align="center"> Optimizing CSAT Through Sentiment Analysis and Predictive Modeling Techniques</div>

## Summary

Customer Satisfaction (CSAT) is a key performance metric in any service-driven organisation. However, CSAT scores often fail to capture the full customer experience, especially when sentiment in textual feedback contradicts the given score.

To address this, we developed an AI-based analytics pipeline that combines **Natural Language Processing (NLP)** and **Machine Learning (ML)** to:

- Automatically extract sentiment from customer comments.

- Align this sentiment with CSAT scores to identify misclassifications or hidden dissatisfaction.

- Predict satisfaction levels from structured support ticket.

The result is a smarter, scalable system for understanding customer satisfaction, enabling better resource planning and faster issue resolution.

🎯 Project Aim:
To enhance customer satisfaction analysis by aligning sentiment from customer feedback with CSAT scores, and to develop accurate predictive models that can forecast CSAT outcomes using Machine Learning techniques.

## Solution Overview

### 🧠 Phase 1: Sentiment Detection from Customer Comments

- Started with AFINN, a rule-based lexicon, which produced inaccurate classifications.

- However, as shown below, it fails to capture the actual sentiment of customer feedback correctly.

![1](images/lexiconresults.png)
  
- Due to AFINN’s limitations, we explored a more advanced BERT model, which significantly improved sentiment detection.
  
- Upgraded to a fine-tuned BERT model, significantly improving sentiment accuracy by understanding context, negation, and domain-specific phrasing.

![2](images/BERTRES.png)

**Result:** A multilingual BERT model fine-tuned on historical feedback data achieved high classification accuracy and contextual understanding.

### 🔁 Phase 2: Aligning Sentiment with CSAT Scores

![3](images/objtwo.png)

- The model detected 5.75% of user-labeled Negative feedback as actually Positive, improving sentiment alignment.

- It also detected 0.65% of user-labeled Positive feedback as actually Negative, uncovering hidden dissatisfaction.

- Business Impact: This alignment improves the credibility of CSAT reporting and helps surface hidden service issues.

### 🧩 Phase 3: Predicting CSAT with Machine Learning

We used structured ticket metadata (country, region, sentiment polarity, etc.) to train multiple classification models:

- Logistic Regression

- Random Forest

- Support Vector Machine (SVM)

- Gradient Boosting Machine (GBM)

All models achieved high accuracy, but **Logistic Regression** performed best on ROC-AUC, highlighting its strength in distinguishing satisfaction levels.

- Evaluation Metrics:

- Accuracy – Overall correctness of predictions.
- ROC-AUC Score – Ability to distinguish sentiment polarity.
- Precision, Recall, F1-score – Balance between false positives & false negatives.
- Confusion Matrix – Insights into correct vs. misclassified instances.

**Key Finding:**

![3](images/mlresults.png)

- Logistic Regression achieved the highest ROC-AUC (0.9512), demonstrating its superior ability to distinguish sentiment polarity, despite high accuracy across all models (97.62%).

##  Summary of Outcomes

✅ BERT Model outperforms traditional methods in interpreting customer sentiment.

✅ Sentiment-CSAT misalignment highlights previously unseen issues.

✅ Predictive ML models enable accurate classification of customer satisfaction.

✅ Logistic Regression offers strong, interpretable performance for operational use.



## Future Enhancements

- Expand to Multilingual Feedback – Incorporate customer reviews in different languages to improve global applicability.

- Feature Expansion – Add Ticket Priority, User Type, and additional metadata for better prediction accuracy.

- Explore Advanced Transformers – Investigate more sophisticated NLP models for improved sentiment classification.


