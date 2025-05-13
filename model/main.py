# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    data = pd.read_csv('data\data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({ 'M' : 1, 'B' : 0 })
    return data

def create_model(data):
    # Create feature matrix and target vector
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f'Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report: \n {classification_report(y_test, y_pred)}')

    return model, scaler

def main():
    # Load data
    data = get_clean_data()

    # Create model
    model, scaler = create_model(data)

    # Export model
    with open('model\model.pkl', 'wb') as f: #(filename, write+binary)
        pickle.dump(model, f)

    with open('model\scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
