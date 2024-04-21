# Seoul Bike Sharing Demand Prediction

This repository contains code for predicting bike sharing demand in Seoul using regression analysis techniques. The dataset includes weather information such as temperature, humidity, wind speed, visibility, dew point temperature, solar radiation, snowfall, rainfall, as well as date and time information. The goal is to develop models that accurately predict the number of bikes rented per hour based on these factors.

## Key Features

- **Data Preprocessing**: The dataset is cleaned and preprocessed to remove irrelevant columns, handle categorical variables, and select data for a specific hour.
- **Exploratory Data Analysis (EDA)**: Visualizations are generated to explore the relationships between weather variables and bike rental counts.
- **Traditional Regression**: Simple and multiple linear regression models are implemented using scikit-learn to predict bike rental counts.
- **Neural Network Regression**: TensorFlow is used to build neural network models for regression, including single-node networks and multi-layer networks with various configurations.
- **Model Evaluation**: Mean Squared Error (MSE) is calculated to evaluate the performance of linear regression and neural network models. Additionally, scatter plots are generated to compare predicted values against true values.
- **Streamlit Application**: A Streamlit web application is included for interactive prediction of bike sharing demand using both linear regression and neural network models.

## Dataset Source

[Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

## Dependencies

- NumPy
- Matplotlib
- Pandas
- scikit-learn
- imbalanced-learn
- Seaborn
- TensorFlow
- Streamlit

## Usage

1. Clone the repository: `git clone https://github.com/shivam-nath-10003/seoul-bike-sharing-prediction.git`
2. Install dependencies.
3. Run the Jupyter notebook or Python scripts to explore the dataset, preprocess the data, build regression models, and evaluate their performance.
4. To run the Streamlit web application:


## Streamlit Application

A Streamlit web application is included for interactive prediction of bike sharing demand using both linear regression and neural network models. The application allows users to visualize predictions based on temperature using either model. Simply run the application using the command mentioned above and follow the instructions in your web browser.

The Streamlit application provides:
- Linear Regression Model: Predictions based on simple linear regression using temperature as the input feature.
- Neural Network Model: Predictions based on a neural network with a single hidden layer using temperature as the input feature.

Explore the predictions by inputting different temperature values and observing the predicted bike counts visually on interactive plots.

## Contributing

Contributions are welcome! Feel free to open issues for bugs, feature requests, or to submit pull requests with enhancements or fixes.
