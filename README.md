# Arabic-Sentiment-Analysis
# Introduction
This project aims to train a bidirectional LSTM model to classify Arabic tweets as positive or negative. The model is trained using TensorFlow and Keras.
# Data
The dataset used for training and testing is the Large Movie Review Dataset, which contains 50,000 movie reviews from IMDB. The dataset is split into 25,000 reviews for training and 25,000 reviews for testing. Each review is labeled as either positive or negative.
You can find the dataset here : https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus
# Preprocessing
The first step in the preprocessing is to create a TensorFlow Keras text dataset for training and validation. The data is split into a training subset and a validation subset using a validation split of 0.2. The TextVectorization layer is used to preprocess the input data. The maximum number of features, embedding dimension, and sequence length are defined, and a custom standardization function is created to preprocess the input data. The TextVectorization layer is then adapted to the text data from the training dataset to generate the vocabulary. Finally, the vectorize_text function is defined to vectorize the text and labels.
# Model
The next step is to define a TensorFlow Keras sequential model. The model consists of an embedding layer, two bidirectional LSTM layers, two dense layers, and a dropout layer. The activation function for the output layer is set to sigmoid.

# Training
The model is trained using the training dataset and validated using the validation dataset. The optimizer used is Adam, and the loss function used is binary cross-entropy. The number of epochs is set to 5
# Results
The model achieves an accuracy of 97% on the test dataset.
# Conclusion
In conclusion, this project demonstrates how to train a bidirectional LSTM model to classify Arabic tweets as positive or negative using TensorFlow and Keras. The code can be easily adapted to other text classification tasks with similar datasets.
