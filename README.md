# Language Classification with Naive Bayes

A machine learning project that classifies text into Slovak (sk), Czech (cs), and English (en) languages using Multinomial Naive Bayes classification with subword tokenization.

# Dataset Structure
Training sentences in three languages (sk, cs, en)

Validation sentences for model evaluation

Data stored in separate files for each language

# Requirements

numpy
matplotlib
scikit-learn
joblib
pickle
tqdm

# Features
Text preprocessing and cleaning

Subword tokenization

Statistical analysis of language data

Zipf's law visualization

Confusion matrix plotting

# Model Components

1.Text Preprocessing

Lowercase conversion

Punctuation and digit removal

Special character handling

Subword splitting

2.Vectorization

CountVectorizer for text-to-feature conversion

Vocabulary building

Subword merging strategy

3.Classification

Multinomial Naive Bayes classifier

Hyperparameter tuning (alpha, fit_prior)

Model evaluation using F1 score

F1 score evaluation

# Usage

1. Data Preparation:
   data_raw = {
    'sk': open_file('train_sentences.sk'),
    'cs': open_file('train_sentences.cs'),
    'en': open_file('train_sentences.en')
}

2. Training:
   naive_classifier = MultinomialNB(fit_prior=False)
   naive_classifier.fit(X_train, y_train)

3. Prediction:
   predictions = naive_classifier.predict(X_val)

# Performance Analysis
Confusion matrix visualization

F1 score calculation

Statistical analysis of language distributions

Vocabulary analysis

# Model Storage
Models saved using joblib

Vectorizers preserved for consistent preprocessing

Merge orders stored in pickle format
