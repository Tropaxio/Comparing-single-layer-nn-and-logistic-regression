import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

# Transform a CSV file into a pandas DataFrame
def csv_to_pd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()

    return df 

# Returns a transformer that one-hot-encodes categorical predictors and standardises numerical ones
def transformer(cat: list[str], num: list[str]):
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    scalar = StandardScaler(with_mean=True, with_std=True)

    Clmtransformer = ColumnTransformer(
        [("categorical", enc, cat),
         ("numerical", scalar, num)],
         remainder='passthrough'
    )

    return Clmtransformer

# Split the data into training and test sets
def make_splits(X, y, test_size, random_state, shuffle):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
        )

    return X_train, X_test, y_train, y_test

# Define random oversampling
def random_oversampling(
    sampling_strategy,
    random_state
):
    return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

# Define random undersampling 
def random_undersampling(
    sampling_strategy,
    random_state
):
    return RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)

# Define SMOTE oversampling 
def smote(
    sampling_strategy,
    random_state,
    k_neighbors
):
    return SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)

# Define Tomek
def tomek(
        sampling_strategy
):
    return TomekLinks(sampling_strategy=sampling_strategy)
    
# Define SMOTE-Tomek hybrid sampling 
def smote_tomek(
    sampling_strategy,
    random_state,
    smote, 
    tomek
):
    return SMOTETomek(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        smote=smote
        tomek=tomek=
    )