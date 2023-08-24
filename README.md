# Retina_Vision
# Welcome to Hybrid Classical-Quantum Neural Network Classifier Android App!

This app allows you to upload an eye image and uses a trained machine learning model to predict whether the patient has normal, severe, or mild diabetes. The model is a hybrid classical-quantum neural network, which combines the power of classical machine learning with the quantum computing capabilities of IBM Quantum Experience.

This project demonstrates how to make API calls to a hybrid classical-quantum neural network classifier using an Android app. The app utilizes the Volley library to communicate with the API, which predicts whether an uploaded eye image is normal, severely diabetic, or mildly diabetic. The model is implemented using a Flask API (`API_Fetching.py`) and the trained classifier is saved as `final_model.pth`.

## Repository Structure

- **[hybrid-classical-quantum-neural-network-classifier](https://github.com/itstechaj/hybrid-classical-quantum-network-classifier):** This repository contains the Python code for the hybrid classical-quantum neural network classifier. 

## Prerequisites

1. Install the required libraries for the Python model by running: `pip install -r requirements.txt`
2. Install Ngrok: `npm install -g ngrok`

## Setting up the API

1. Navigate to the `Inference using Flask API Folder` repository.
2. Run the Flask API using the command: `python API_Fetching.py`
3. Expose the API using ngrok: `ngrok http port_number`

Note: You'll need to authenticate with ngrok to obtain the exposed URL.To test the API locally, we use Ngrok to create a secure tunnel between our local machine and the internet. 

## Android App Setup

1. Open the Android app project in Android Studio.
2. Make sure you have added the necessary dependencies, including the Volley library.
3. In the app's Java code, update the API endpoint with the ngrok URL obtained earlier.

## Making Predictions

1. Launch the Android app on a device or emulator.
2. Navigate to the prediction section.
3. Click the 'Upload' button to select an eye image.
4. After selecting the image, click the 'Predict' button.
5. The app will send the image to the Flask API using the Volley library.
6. The API will process the image through the hybrid classical-quantum neural network classifier.
7. The predicted classification (normal, severe diabetic, mild diabetic) will be sent back to the app.
8. The app will display the prediction on the screen.

## API Fetching and Preprocessing Code

The `API_Fetching.py` file contains the Flask API code responsible for receiving image data and performing preprocessing before passing it through the trained model. Image preprocessing involves cropping, resizing, drawing a circle, and applying Gaussian blur to enhance model performance.

Please note that the preprocessing code is adapted to your project's requirements. You can further modify it as needed.

## Conclusion

This project showcases the integration of a hybrid classical-quantum neural network classifier with an Android app. By following the steps outlined in this README, you can successfully upload an eye image through the app, have it processed by the classifier via the Flask API, and receive and display the prediction within the app interface. This provides a practical example of how machine learning models can be deployed and accessed through mobile applications.

## Contributing

Contributions are welcome! Please open a pull request with your proposed changes, and we will review them promptly.

## License
This project is licensed under the MIT License.

