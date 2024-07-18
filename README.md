# F1 Driver Position Prediction

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project aims to predict the final positions of Formula 1 drivers using machine learning models. The main goal is to provide insights and predictions that can be utilized for strategic decisions in races. By leveraging historical race data, the project builds predictive models that forecast driver positions based on various race features.

## Dataset

The dataset used for this project consists of historical Formula 1 race data, including driver information, constructor details, race circuits, and race results. The data includes:

- Race details (year, date, circuit)
- Driver details (name, constructor)
- Race performance metrics (grid position, laps completed, final rank)

The dataset is stored in a CSV file named `F1.csv` under data folder.

## Model

Several machine learning models were trained and evaluated for this project:

- **Dummy Classifier**: Used as a baseline model with an average accuracy of 34.94%.
- **Random Forest Classifier**: Achieved an average accuracy of 60.15%.
- **CatBoost Classifier**: Achieved an average accuracy of 59.74%.
- **LightGBM Classifier**: Achieved an average accuracy of 59.69%.

The Random Forest Classifier was chosen as the final model due to its simplicity and lower computational resource requirements compared to CatBoost and LightGBM.

## Installation

To run this project, ensure you have the following dependencies installed:

```sh
pip install pandas numpy scikit-learn scipy catboost lightgbm
```

## Usage

### Predictor Function

The `predictor_grupos` function is the core function used to predict driver positions. Here is how to use it:

1. **Load the Model**:
    ```python
    import pickle
    import pandas as pd

    model_path = '../Modelo/best_rf_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    ```

2. **Load the Dataset**:
    ```python
    df = pd.read_csv('../Test/Test_F1.csv')
    ```

3. **Predict and Save Results**:
    ```python
    resultados = predictor_grupos(model_path, df)

    # Save results to CSV
    for carrera, resultado in resultados.items():
        filename = f'results_{carrera}.csv'
        resultado['result_df'].to_csv(filename, index=False)
    ```

### Web Application

A prototype web application can be created using Streamlit to provide an interactive interface for predictions. Users can select race data and make predictions for specific drivers or all drivers in a race.

1. **Install Streamlit**:
    ```sh
    pip install streamlit
    ```

2. **Run the Streamlit Application**:
    ```sh
    streamlit run app.py
    ```
    
## Results

The model's predictions have shown significant improvements over the baseline, providing a reliable tool for forecasting driver positions in races. The Random Forest Classifier's robustness and efficiency make it the preferred choice for deployment.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure you adhere to the project's coding standards and include appropriate tests.
