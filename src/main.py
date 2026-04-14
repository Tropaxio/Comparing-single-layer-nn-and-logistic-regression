import os 
from dotenv import load_dotenv 
import data 
import metrics 
from models import logistic_regression, neural_network


# This is the main function 
def main():

    # Load the path as an environment variable
    load_dotenv()
    path = os.getenv("CSV_PATH")  

    # Transform the CSV into a Pandas df
    Default = data.csv_to_pd(path)
    
    # Define numerical and categorical predictors, and response variable
    X = Default.drop(columns=["default"])
    y = Default["default"]
    cat = X["student"]
    num = X[["balance", "income"]]



if __name__ == '__main__':
    main()