import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['Activity_level'] = LabelEncoder().fit_transform(df['Activity_level'])

    features = df.drop('Calories', axis=1)
    labels = df['Calories']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, labels