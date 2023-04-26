# CIS4496 Honors Contract Project Source Code:

This folder hosts the code for the project.

### Files/Folders:

- `Tutorial.ipynb` - Python notebook for my version of tutorial code provided by Kaggle Competition

- `LogisticRegression.ipynb` - My first run of whole process for competition using a Logistic Regression model. Includes partial EDA, preprocessing, modeling, and evaluation.

- `EDA.ipynb` - EDA code and charts for final model. More can be viewed on Kaggle Submission

- `Preprocess.ipynb` - Preprocessing code for final model.

- `Model.ipynb` - BERT Model code for final model as well as evaluation of results

- `Evaluation.ipynb` - Old file for carrying evaluation metrics of BERT model. To be deleted.

- `tokenization.py` - tokenization code from TensorFlow [https://github.com/tensorflow/models/blob/master/official/nlp/tools/tokenization.py]. Tokenization is used in natural language processing to split paragraphs and sentences into smaller units that can be more easily assigned meaning.
Done using FullTokenizer class from tensorflow/models/official/nlp/bert/tokenization.py

- `bertModel.py` - Using an implementation of BERT from TensorFlow Models: tensorflow/models/official/nlp/bert [https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1]. 
Bidirectional Encoder Representations from Transformers is a family of masked-language models which is used to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context
