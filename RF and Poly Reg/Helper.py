import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Loading the data.
def load_data(dataset = 'Linear', folder_path = '../Datasets/Final Datasets/'):
    data_path = folder_path + str(dataset) + '.csv'

    if os.path.isfile(data_path):
        data = pd.read_csv(data_path)
    else:
        print("Path Not Found")

    return data

# Processing the data, including removing columns.
def proc_data(dataframe, del_columns = []):
    processed_data = dataframe.drop(del_columns, axis = 1)

    return processed_data

# Split the data into x, y, train and test.
def split_data(data_set, target_var, ratio = 0.7):
    X = data_set.drop(target_var, axis = 1)
    Y = data_set[target_var]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 1 - ratio, random_state=19)
    return x_train, x_test, y_train, y_test

# GridSearch to find the best parameters (uses Cross Validation)
def Grid_Search(model, x_train, y_train, x_test, y_test, parameters, folds = 5, scores = 'neg_mean_squared_error'):
    fine_tune_model = GridSearchCV(model, parameters, cv = folds, scoring = scores, verbose = 3)
    fine_tune_model.fit(x_train, y_train)

    print("Best Parameters:", fine_tune_model.best_params_)
    print("Best Cross-validation Score:", fine_tune_model.best_score_)

    y_pred = fine_tune_model.best_estimator_.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE -> {mse}")
