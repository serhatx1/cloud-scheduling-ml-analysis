import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

def train():
    df = pd.read_csv("cloud_workload_dataset.csv")
    
    # 1. Preprocessing
    # Drop IDs and Timestamps
    df = df.drop(['Job_ID', 'Task_Start_Time', 'Task_End_Time'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Data_Source', 'Job_Priority', 'Resource_Allocation_Type']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Feature scaling for numerical columns
    X = df.drop('Scheduler_Type', axis=1)
    y = df['Scheduler_Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Define Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    # 3. Train and Evaluate
    print("--- Model Training and Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-Score": f1_score(y_test, y_pred, average='weighted')
        }
        print(f"{name} Results: {results[name]}")

    # 4. Save results to JSON for LaTeX reporting
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining completed. Results saved to 'model_results.json'.")

if __name__ == "__main__":
    train()
