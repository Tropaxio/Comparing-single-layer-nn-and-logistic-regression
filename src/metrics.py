from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# Function that computes the confusion matrix
def conf_matrix(
        y_true,
        y_pred
):
    return confusion_matrix(
        y_true=y_true,
        y_pred=y_pred
    )

# Function that computes the accuracy score
def acc_score(
        y_true, 
        y_pred
):
    return accuracy_score(
        y_true=y_true,
        y_pred=y_pred
    )

def rec_score(
        y_true,
        y_pred
):
    return recall_score(
        y_true=y_true,
        y_pred=y_pred 
    )