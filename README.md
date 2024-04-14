
# AutoEncoder for Movie Recommendations

This project implements a Stacked AutoEncoder (SAE) for movie recommendations using PyTorch. The SAE is trained on the MovieLens 100k dataset to learn the underlying patterns in user-movie interactions and predict ratings for unrated movies.

## Dataset

The MovieLens 100k dataset is used for training and testing the SAE. It consists of two main files:

- `u1.base`: Training set containing user-movie interactions.
- `u1.test`: Test set for evaluating the performance of the trained SAE.

## Requirements

Make sure you have the following libraries installed:

- numpy
- pandas
- torch

## Usage

1. Clone the repository.
2. Download the MovieLens 100k dataset and place it in the appropriate directory (`ml-100k/ml-100k/`).
3. Run the `AutoEncoder.py` script to train the SAE.
4. Evaluate the SAE performance using the test set.

## Training

The SAE architecture consists of multiple fully connected layers:

- Input layer: Number of movies
- Hidden layers: 30, 20, and 10 neurons, respectively
- Output layer: Same size as input layer

The model is trained using the RMSprop optimizer with a learning rate of 0.01 and L2 regularization (weight decay = 0.5).

## Results

The training process involves multiple epochs of forward and backward passes. After training, the SAE can predict ratings for unrated movies based on user preferences learned during training.

## Testing

The trained SAE is evaluated using the test set to measure its performance in predicting ratings for unseen user-movie pairs.

---
