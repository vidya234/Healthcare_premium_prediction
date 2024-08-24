import pandas as pd
import joblib

# Load pre-trained models and scalers
model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    """
    Calculate the normalized risk score based on the medical history.

    Args:
        medical_history (str): The medical history of the individual.

    Returns:
        float: The normalized risk score.
    """
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Convert medical history to lowercase and split into diseases
    diseases = medical_history.lower().split(" & ")

    # Sum the risk scores for each disease
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)

    max_score = 14  # Maximum possible risk score
    min_score = 0   # Minimum possible risk score

    # Normalize the risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def preprocess_input(input_dict):
    """
    Preprocess the input dictionary into a DataFrame suitable for model prediction.

    Args:
        input_dict (dict): The input data dictionary.

    Returns:
        pd.DataFrame: DataFrame with preprocessed features.
    """
    # Define the expected columns for the DataFrame
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Assign values based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Calculate normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Scale the features
    df = handle_scaling(input_dict['Age'], df)

    return df

def handle_scaling(age, df):
    """
    Scale the features in the DataFrame based on the age using the appropriate scaler.

    Args:
        age (int): The age of the individual.
        df (pd.DataFrame): DataFrame with features to be scaled.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    # Choose the scaler based on age
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # Add dummy column to match the expected scaler input
    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Drop the dummy column
    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    """
    Predict the outcome based on the input dictionary.

    Args:
        input_dict (dict): The input data dictionary.

    Returns:
        int: The predicted value.
    """
    input_df = preprocess_input(input_dict)

    # Choose the model based on age
    model = model_young if input_dict['Age'] <= 25 else model_rest

    # Make the prediction
    prediction = model.predict(input_df)

    return int(prediction[0])
