
# Next Word Predictor using LSTM

This repository contains a deep learning model built using TensorFlow and Keras for predicting the next word in a given sentence. The model utilizes a Long Short-Term Memory (LSTM) network and is trained on a dataset of English text to predict the next word based on the previous words.

## Table of Contents

* [Introduction](#introduction)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training the Model](#training-the-model)
* [Prediction](#prediction)


## Introduction

The goal of this project is to build a model that can predict the next word in a given sentence using an LSTM-based architecture. The model is trained on a text corpus, tokenizes the text, and builds a sequence prediction task. It uses a sequence of words as input and predicts the next word in the sequence.

## Setup and Installation

To run this project, you need to have the following libraries installed:

* **TensorFlow**: For building and training the LSTM model.
* **Keras**: For high-level neural networks API.
* **NumPy**: For numerical operations.
* **Pandas**: Optional, for dataset management.
* **Matplotlib**: Optional, for visualizing results.

To install the required libraries, you can use the following command:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

### 1. Loading Data

The dataset is read from a text file. The text is split into sentences, and each sentence is tokenized, where each unique word is assigned a unique index using the `Tokenizer` from Keras.

### 2. Tokenization and Feature Engineering

The sentences are tokenized and transformed into numerical sequences. The input sequences are created by adding one word at a time, building a feature set for supervised learning.

### 3. Padding the Sequences

To ensure that all input sequences have the same length, padding is added to the sequences with zeros, making them uniform for model input.

### 4. One-hot Encoding

The labels are one-hot encoded to represent the probability distribution over all words in the vocabulary. This enables the model to predict the likelihood of each word as the next word.

### 5. Model Architecture

The model consists of:

* **Embedding Layer**: Converts words into dense vectors.
* **LSTM Layer**: Captures sequential dependencies between words.
* **Dense Layer**: Outputs a probability distribution for each word in the vocabulary.

### 6. Saving and Loading the Model

After training, the model can be saved and loaded for future use. This allows the model to be used for prediction without retraining.

### 7. Making Predictions

To predict the next word in a sequence, a starting sentence is passed to the model. The model iteratively predicts the next word and appends it to the sentence until the desired sequence length is achieved.

## Model Architecture

The model consists of the following layers:

* **Embedding Layer**: This layer converts words into dense vectors of fixed size, allowing the model to learn meaningful relationships between words.
* **LSTM Layer**: A recurrent neural network layer that helps the model learn from sequences of words, capturing long-range dependencies in the data.
* **Dense Layer**: This layer generates a probability distribution over the vocabulary, helping to predict the most likely next word based on the input sequence.

## Training the Model

The model is trained using an optimizer (such as Adam) and a loss function (such as sparse categorical crossentropy). The training process may take several epochs, depending on the size of the dataset and the computational resources available.

## Prediction

Once the model is trained, it can predict the next word in a sentence. Given an initial text input, the model predicts the next word, and this process is repeated iteratively to generate a sequence of words. The prediction is based on the learned relationships between words in the training data.
