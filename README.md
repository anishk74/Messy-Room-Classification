# Rooms-classification
Detect if a room is messy with the help of CNN.

## Description
The dataset contains limited image data of messy and clean rooms. 

## Technologies
* Python3
  * TensorFlow
  * NumPy
  * Pandas
  * Seaborn
  * Matplotlib

## Initial Histogram of image data

![](plots/init_hist.png?raw=true "Title")

## Histogram of training data after data augmentation

Used data augmentation to increase the number of training images which helps the model to learn better.
![](plots/aug_hist.png?raw=true "Title")

## Performance on Cross-Validation data

![](plots/output_measures.png?raw=true "Title")

The plot shows the Training accuracy and Cross-Validation accuracy throughout the training epochs.