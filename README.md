# Churn Prediction Using Artificial Neural Networks

This project involves building and training an Artificial Neural Network (ANN) to predict customer churn based on a given dataset. Customer churn refers to the likelihood of customers discontinuing their relationship with a company or service. By predicting churn, businesses can take proactive measures to retain valuable customers.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Project Description

The goal of this project is to develop a machine learning model using an Artificial Neural Network (ANN) to accurately predict whether a customer will churn. The model is trained on a dataset containing various customer-related features such as tenure, balance, number of products, and more.

## Dataset

The dataset used for this project is `Churn_Modelling.csv`, which contains the following features:
- **CustomerId**: Unique identifier for each customer.
- **Surname**: The surname of the customer.
- **CreditScore**: The credit score of the customer.
- **Geography**: The country from which the customer belongs.
- **Gender**: The gender of the customer.
- **Age**: The age of the customer.
- **Tenure**: The number of years the customer has been with the bank.
- **Balance**: The account balance of the customer.
- **NumOfProducts**: The number of products the customer has with the bank.
- **HasCrCard**: Whether the customer has a credit card (1 for yes, 0 for no).
- **IsActiveMember**: Whether the customer is an active member (1 for yes, 0 for no).
- **EstimatedSalary**: The estimated salary of the customer.
- **Exited**: The target variable, indicating whether the customer has churned (1) or not (0).

## Model Architecture

The ANN model consists of the following layers:
1. **Input Layer**: Corresponds to the features of the dataset.
2. **Hidden Layers**: Several fully connected layers with ReLU activation.
3. **Output Layer**: A single neuron with sigmoid activation to predict the probability of churn.

## Dependencies

The following libraries are required to run the notebook:
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install these dependencies using pip:

```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/churn-ann.git
    ```

2. Navigate to the project directory:

    ```bash
    cd churn-ann
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook:

    ```bash
    jupyter notebook artificial_neural_network.ipynb
    ```

## Usage

1. Load the dataset `Churn_Modelling.csv` into the notebook.
2. Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
3. Train the ANN model using the preprocessed data.
4. Evaluate the model's performance on the test set.
5. Use the trained model to predict customer churn on new data.

## Results

The trained ANN model will output the probability of each customer churning. The performance of the model can be evaluated using metrics such as accuracy, precision, recall, and the F1-score. The final results will provide insights into which features are most indicative of customer churn and how well the model can predict churn.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.
