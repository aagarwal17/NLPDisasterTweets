# CIS4496 Honors Contract Project Data:

This folder hosts the datasets for the project.

Data provided by [Kaggle Competition](https://www.kaggle.com/competitions/nlp-getting-started/data)

### Files/Folders:

- `train.csv` - original training data provided by competition. Contains:
   - *id*: identifier for each tweet
   - *text*: the tweet itself
   - *keyword*: keyword from tweet (many are blank)
   - *location*: location the tweet was sent from (many are blank)
   - *target*: disaster (1) or not (0)

- `test.csv` - original testing data provided by competition. Contains same columns as training dataset except the target

- `sample_submission.csv` - sample submission file, for formatting

- `df_train_cleanedMislabelsDuplicates.csv` - produced training dataset after preprocessing with mislabels and duplicates kept in

- `df_train_cleanedMislabelsNoDuplicates.csv` - produced training dataset after preprocessing with mislabels kept in and duplicates removed

- `df_train_cleanedNoMislabelsDuplicates.csv` - produced training dataset after preprocessing with mislabels removed and duplicates kept in

- `df_train_cleanedNoMislabelsNoDuplicates.csv` - produced training dataset after preprocessing with mislabels and duplicates removed

- `df_test_cleaned.csv` - produced test dataset after preprocessing
