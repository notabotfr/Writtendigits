# MNIST Handwritten Digit Recognition

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset and predict digits from user-provided images. It includes data loading, preprocessing, model building, training, evaluation, and a prediction pipeline for real-world images.

## Overview

The MNIST dataset is a classic benchmark for image classification tasks. This project leverages the power of CNNs to achieve high accuracy in classifying handwritten digits. Furthermore, it adds functionality to predict digits from external images through image processing and prediction.

## Dataset

The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is a 28x28 grayscale image.


## Libraries Used

*   **numpy:** For numerical computations.
*   **keras:** For building and training the neural network model.
*   **matplotlib:** For data visualization and image display.
*   **opencv-python (cv2):** For image preprocessing (thresholding, resizing, etc.).
*   **PIL (Pillow):** For opening and manipulating images.
*   **scikit-learn (sklearn):** For performance metrics like classification report and confusion matrix

## Implementation

The Jupyter Notebook (`MNIST_Digit_Recognition.ipynb`) contains the following steps:

1.  **Data Loading and Preprocessing:**
    *   Loading the MNIST dataset using `keras.datasets.mnist.load_data()`.
    *   Reshaping the input images to the appropriate format for the CNN.
    *   Normalizing pixel values to the range [0, 1].
    *   Converting the labels to one-hot encoded vectors using `keras.utils.to_categorical()`.

2.  **Model Building:**
    *   Creating a CNN model with convolutional layers, max pooling layers, dropout layers, and dense layers.
    *   Using 'relu' activation for hidden layers and 'softmax' activation for the output layer.
    *   Compiling the model with the Adam optimizer and categorical cross-entropy loss.

3.  **Model Training:**
    *   Training the model using `model.fit()` with a validation split and callbacks for early stopping and learning rate reduction.
    *   Monitoring 'val_loss' for early stopping and 'val_accuracy' for ReduceLROnPlateau.

4.  **Model Evaluation:**
    *   Evaluating the model's performance on the test set using `model.evaluate()`.
    *   Generating predictions on the test set.
    *   Visualizing the training history (accuracy and loss curves).
    *   Displaying the confusion matrix and classification report.

5.  **Digit Prediction from External Images:**
    *   `preprocess_image(image_path)`: Loads, preprocesses, resizes, and normalizes external images.
    *   `predict_digit(image_path, model)`: Predicts digit from external images using the trained model.
    *   `process_and_predict(image_path, model)`:  Processes the image, makes a prediction, and shows the results, along with the original and preprocessed images. This improved function uses the refactored preprocessing and prediction functions.

## Running the Code

1.  **Install Dependencies:**
    ```bash
    pip install numpy keras matplotlib opencv-python pillow scikit-learn
    ```
    Make sure you have tensorflow or pytorch installed according to your keras backend
    ```bash
    pip install tensorflow
    #OR
    pip install torch
    ```

2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook MNIST_Digit_Recognition.ipynb
    ```

3.  **Run the Notebook:** Execute the cells in the notebook sequentially to train the model, evaluate it, and test the prediction pipeline.

4.  **Test with Custom Images (Optional):**
    *   Place an image of a handwritten digit (e.g., `example_image.jpg`) in the same directory as the notebook.
    *   Modify the `image_path` variable in the notebook to point to your image.
    *   Run the `process_and_predict()` function to see the model's prediction.

## Function Descriptions

- **`preprocess_image(image_path)`:** Preprocesses the image including reading the image, resizing it, converting it to grayscale, normalizing pixel values, and reshaping it for the model.
- **`predict_digit(image_path, model)`:** Predicts the digit in the image using the preprocessed image and returns the predicted digit and the confidence.
- **`display_results(original_image, preprocessed_image, predicted_digit, confidence)`:** Displays the original image, the preprocessed image, predicted digit, and its confidence.
- **`process_and_predict(image_path, model)`:** Orchestrates the image prediction process, invoking the above functions.

## Results

*   The CNN model achieves high accuracy (around 99%) on the MNIST test set.
*   The training history visualizations show the model's learning progress.
*   The confusion matrix provides insights into the model's performance on different digit classes.
*   The `process_and_predict()` function demonstrates the ability to predict digits from external images successfully.

## Notes

*   The performance of the model can be further improved by experimenting with different CNN architectures, hyperparameters, and training techniques.
*   Image preprocessing steps are crucial for achieving good results with real-world images.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to submit a pull request.

## License

[MIT License](LICENSE)
