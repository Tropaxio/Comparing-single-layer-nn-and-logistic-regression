import os 
from dotenv import load_dotenv 
import data 
import metrics 
from models.logistic_regression import logistic_regression



# This is the main function 
def main():

    # Load the path as an environment variable
    load_dotenv()
    path = os.getenv("CSV_PATH")  

    # Transform the CSV into a Pandas df
    Default = data.csv_to_pd(path)
    
    # Define numerical and categorical predictors, and response variable
    X = Default.drop(columns=["default"])
    y = Default["default"].map({"No": 0, "Yes": 1})
    cat = ["student"]
    num = ["balance", "income"]

    # Separate the data into training and test sets
    X_train, X_test, y_train, y_test = data.make_splits(
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    # Apply the transformations
    ct = data.transformer(cat=cat, num=num)
    X_train_transf = ct.fit_transform(X_train)
    X_test_transf = ct.transform(X_test)

    # Apply logistic regression
    clf = logistic_regression(fit_intercept= True, max_iter=200)
    clf.fit(X_train_transf, y_train)

    y_hat = clf.predict(X_test_transf)
    
    




    


if __name__ == '__main__':
    main()