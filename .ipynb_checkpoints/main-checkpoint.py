import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def main():
    # Load the Iris dataset
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    
    # Prepare the data
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model to a file
    with open('iris_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model training and saving completed.")

if __name__ == "__main__":
    main()
