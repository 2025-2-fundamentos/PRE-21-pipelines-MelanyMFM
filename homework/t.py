import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np 


def load_data():
    """Replicates the load_data function from the test script."""
 
    dataframe = pd.read_csv(
        "../files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )
    data = dataframe.phrase
    target = dataframe.target
    return data, target


def create_and_train_estimator(data, target):
 
    estimator = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 

            min_df=5,
            max_df=0.9
        )),
        
        ('clf', LogisticRegression(

            C=10.0, 
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )),
    ])

    # Train the pipeline
    estimator.fit(data, target)
    return estimator


def save_estimator(estimator):
    """Saves the trained model to the exact path required by the test."""
    output_dir = "homework"
    output_path = os.path.join(output_dir, "estimator.pickle")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(estimator, file)
    print(f"✅ Trained estimator saved to {output_path}")


if __name__ == "__main__":
    print("Starting model training with optimized parameters...")
    try:
        data, target = load_data()
        
        # Train Model with improved settings
        trained_estimator = create_and_train_estimator(data, target)
        
        # Save the new, higher-accuracy model
        save_estimator(trained_estimator)
        
        # Verification
        accuracy = accuracy_score(
            y_true=target,
            y_pred=trained_estimator.predict(data),
        )
        print(f"New achieved accuracy on training data: {accuracy:.6f}")

    except FileNotFoundError:
        print("\nFATAL ERROR: The input data file 'files/input/sentences.csv.zip' was not found.")