import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def calculate_velocity_and_acceleration(df):
    df['velocity'] = np.sqrt(np.diff(df['X'])**2 + np.diff(df['Y'])**2)
    df['acceleration'] = np.diff(df['velocity'])
    df = df.iloc[1:].reset_index(drop=True)
    return df

def calculate_angular_momentum(df, person_idx):
    mass = 1
    moment_of_inertia = 1
    angular_momentum = []
    for i in range(1, len(df)):
        x1, y1 = df.loc[i-1, 'X'], df.loc[i-1, 'Y']
        x2, y2 = df.loc[i, 'X'], df.loc[i, 'Y']
        r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angular_momentum.append(mass * r * moment_of_inertia)
    df['angular_momentum'] = angular_momentum
    df = df.iloc[1:].reset_index(drop=True)
    return df

def extract_features(csv_file):
    df = pd.read_csv(csv_file)
    df = calculate_velocity_and_acceleration(df)
    df = calculate_angular_momentum(df, person_idx=0)
    features = df[['velocity', 'acceleration', 'angular_momentum']].values
    return features

def load_data_from_folder(folder_path):
    X, y = [], []
    for csv_file in os.listdir(folder_path):
        if csv_file.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, csv_file)
            features = extract_features(csv_file_path)
            label = int(csv_file.split('_')[-1].split('.')[0])
            X.append(features.flatten())
            y.append(label)
    return np.array(X), np.array(y)

csv_folder = './csv_outputs'
X, y = load_data_from_folder(csv_folder)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=41)

svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))

