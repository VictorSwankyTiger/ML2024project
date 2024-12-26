from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

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
            label = int(csv_file.split('_')[-1].split('.')[0])  # Assuming label is in file name
            X.append(features.flatten())
            y.append(label)
    return np.array(X), np.array(y)

csv_folder = './csv_outputs'
X, y = load_data_from_folder(csv_folder)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
svm = SVC(kernel='rbf')

grid_search = GridSearchCV(svm, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

svm_best = grid_search.best_estimator_
y_pred = svm_best.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

