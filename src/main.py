import os 
from dotenv import load_dotenv 
import data 
import metrics 
from models.logistic_regression import logistic_regression
from models import neural_network

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

    # Compute Prediction
    y_hat = clf.predict(X_test_transf)

    # Get confusion matrix and recall score
    matrix = metrics.conf_matrix(y_true=y_test, y_pred=y_hat)
    rec_score = metrics.rec_score(y_true=y_test, y_pred=y_hat)

    ###########################
    
    # Apply the Neural Network
    X_shape = neural_network.get_column(X)

    default_model = neural_network.DefaultModel(X_shape)
    summary = neural_network.get_summary(
        model=default_model,
        input_size=X_train_transf.shape,
        col_names=['input_size', 'output_size', 'num_params']
    )

    # Transform the dataset into tensors and torch dataset
    X_train_tensor = neural_network.transform_to_torch_tensor(X_train_transf)
    y_train_tensor = neural_network.transform_to_torch_tensor(y_train, is_target=True)
    default_train = neural_network.transform_to_torch_dataset(X_train_tensor, y_train_tensor)

    X_test_tensor = neural_network.transform_to_torch_tensor(X_test_transf)
    y_test_tensor = neural_network.transform_to_torch_tensor(y_test, is_target=True)
    default_test = neural_network.transform_to_torch_dataset(X_test_tensor, y_test_tensor)

    # Transform into a pytorch's DataLoader
    default_train_dataloader = neural_network.transform_to_dataloader(
        default_train,
        batch_size=32
    )
    default_test_dataloader = neural_network.transform_to_dataloader(
        default_test,
        batch_size=32
    )

    # Define and fit the trainer
    criterion, optimizer = neural_network.get_training_components(
    model=default_model,     
    lr=0.001
    )

    trainer = neural_network.Trainer(
        model=default_model,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.fit(
        dataloader=default_train_dataloader,
        epochs=200
    )

    evals = neural_network.evaluate(
        model=default_model,
        X=X_test_tensor,
        y=y_test_tensor
    )

    print(evals)   
    
if __name__ == '__main__':
    main()
