# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# Define a function to load and preprocess the data
def load_data(filename):
    # Read the excel file
    data = pd.read_excel(filename)
    # Filter out rows with age less than or equal to 10
    data = data[data['age'] > 10]
    # Fill missing values with 0
    data = data.fillna(0)
    # Create a binary target variable based on the minimum OGTT value
    data['y'] = (data['OGTT'] == 0)
    # Return the data
    return data


# Define a function to split the data into train and test sets
def split_data(data, test_size, random_state):
    # Drop the columns that are not relevant for the prediction
    X = data.drop(['y', 'ID', 'OGTT', 'EFW>90', 'AFI>25', 'AC>90'], axis=1)
    y = data['y']
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Return the train and test sets
    return X_train, X_test, y_train, y_test


# Define a function to scale the data using StandardScaler
def scale_data(X_train, X_test):
    # Create a StandardScaler object
    scaler = StandardScaler()
    # Fit the scaler on the train set
    scaler.fit(X_train)
    # Transform the train and test sets
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # Return the scaled sets
    return X_train_scaled, X_test_scaled


def set_results(X_train, clf, y_train):
    pred = 1 - clf.predict_proba(X_train)
    # Create a dataframe with the true and predicted values
    res = pd.DataFrame(y_train)
    res['pred'] = pred[:, 0]
    # Sort the dataframe by the predicted values
    res = res.sort_values(by='pred')
    # Calculate the cumulative sum of true positives
    res['cumsum'] = res['y'].cumsum()
    # Calculate the false positives by subtracting the cumulative sum from the maximum sum
    res["FP"] = res['cumsum'].max() - res['cumsum']
    # Calculate the false positive rate by dividing by the number of samples
    res['FP_percent'] = res["FP"] / res.shape[0]
    # Plot the false positive rate against the predicted values
    plt.plot(res['pred'], res['FP_percent'])
    plt.xlabel('Predicted Probability')
    plt.ylabel('False Positive Rate')
    plt.show()
    # Find the threshold that corresponds to a false positive rate of 0.1 or less
    thresh = (res[res['FP_percent'] <= 0.1]).iloc[0]['pred']
    return thresh


# Define a function to fit and evaluate a logistic regression model
def logistic_regression(X_train, y_train):
    # Create a logistic regression object with a fixed random state and maximum iterations
    clf = LogisticRegression(random_state=0, max_iter=5000)
    # Fit the model on the train set
    clf.fit(X_train, y_train)
    thresh = set_results(X_train, clf, y_train)
    return clf, thresh


# Define a function to fit and evaluate a logistic regression model with regularization
def logistic_regression_reg(X_train, y_train):
    # Create a logistic regression object with a fixed random state and maximum iterations
    clf = LogisticRegression(random_state=0, max_iter=5000)
    # Define the grid of parameters for regularization
    param_grid = {'penalty': ['l1', 'l2'],  # type of regularization
                  'C': [0.01, 0.1, 1, 10, 100]  # inverse of regularization strength
                  }
    # Create a GridSearchCV object to find the optimal values of the parameters
    grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=10)
    # Fit the grid on the train set
    grid.fit(X_train, y_train)
    # Print the best parameters and the best score
    print(grid.best_params_)
    print(grid.best_score_)
    # Get the best estimator from the grid
    clf = grid.best_estimator_
    # Predict the probabilities on the train set
    thresh = set_results(X_train, clf, y_train)
    # Return the model and the threshold
    return clf, thresh


# Define a function to perform bootstrap sampling and calculate the standard errors of the coefficients
def bootstrap_sampling(X_train, y_train, clf, n_bootstraps):
    # Create an empty list to store the bootstrapped coefficients
    bootstrapped_coefs = []
    # Set a random seed for reproducibility
    rng = np.random.RandomState(42)
    # Loop over the number of bootstraps
    for i in range(n_bootstraps):
        print(i)
        # Generate random indices with replacement
        indices = rng.randint(0, len(y_train), len(y_train))
        # Subset the train set with the random indices
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]
        # Fit the model on the bootstrapped set
        clf.fit(X_boot, y_boot)
        # Append the coefficients to the list
        bootstrapped_coefs.append(clf.coef_[0])
    # Convert the list to a numpy array
    bootstrapped_coefs = np.array(bootstrapped_coefs)
    # Calculate the standard errors by taking the standard deviation along the rows
    standard_errors = np.std(bootstrapped_coefs, axis=0)
    # Return the standard errors
    return standard_errors


# Define a function to plot the odds ratios and confidence intervals of the features
def plot_odds_ratios(X_train, clf, standard_errors):
    # Calculate the odds ratios by exponentiating the coefficients
    odds_ratios = np.exp(clf.coef_[0])
    # Plot the odds ratios and the confidence intervals using error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(x=(2 - odds_ratios), y=X_train.columns, xerr=standard_errors, fmt='o')
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.xlabel('Odds Ratio')
    plt.ylabel('Feature')
    plt.title('Odds Ratio Plot for Logistic Regression')
    plt.show()

    # Define a function to plot the ROC curve and calculate the AUC score


def plot_roc_curve(X_test, y_test, clf):
    # Predict the probabilities on the test set
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    # Calculate the false positive rate and the true positive rate
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    # Calculate the area under the curve
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    # Plot the ROC curve and label it with the AUC score
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()

    # Main script


if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data("/home/guylu/Downloads/GCT.xlsx")
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)
    # Scale the data using StandardScaler
    X_train, X_test = scale_data(X_train, X_test)
    # Fit and evaluate a logistic regression model
    # clf, thresh = logistic_regression(X_train, y_train)
    clf, thresh = logistic_regression_reg(X_train, y_train)
    # Perform bootstrap sampling and calculate the standard errors of the coefficients
    standard_errors = bootstrap_sampling(X_train, y_train, clf, n_bootstraps=2000)
    # Plot the odds ratios and confidence intervals of the features
    plot_odds_ratios(X_train, clf, standard_errors)
    # Plot the ROC curve and calculate the AUC score
    plot_roc_curve(X_test, y_test, clf)
    # Print a message to indicate the end of the script
    print("done")

#
#
# import pandas as pd
#
# data = pd.read_excel("/home/guylu/Downloads/GCT.xlsx")
#
# data = data[data['age']>10]
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
#
# data['y'] = (data['OGTT'] == data['OGTT'].min())
# import numpy as np
# from sklearn.model_selection import train_test_split
# data = data.fillna(0)
# X_train, X_test, y_train, y_test = train_test_split(data.drop(['y', 'ID', 'OGTT', 'EFW>90', 'AFI>25', 'AC>90'], axis=1),
#                                                     data['y'], test_size=0.20, random_state=42)
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0,max_iter=5000).fit(X_train, y_train)
# pred = 1 - clf.predict_proba(X_train)
# res = pd.DataFrame(y_train)
# res['pred'] = pred[:,0]
# res = res.sort_values(by='pred')
# res['cumsum'] = res['y'].cumsum()
# res["FP"] = res['cumsum'].max() - res['cumsum']
# res['FP_percent'] = res["FP"]/res.shape[0]
# import matplotlib.pyplot as plt
# plt.plot(res['pred'], res['FP_percent'])
# plt.show()
#
# thresh = (res[res['FP_percent'] <= 0.1]).iloc[0]['pred']
#
#
#
# n_bootstraps = 2000
# bootstrapped_coefs = []
# rng = np.random.RandomState(42)
# for i in range(n_bootstraps):
#     print(i)
#     indices = rng.randint(0, len(y_train), len(y_train))
#     X_boot = X_train.iloc[indices]
#     y_boot = y_train.iloc[indices]
#     clf_boot.fit(X_boot, y_boot)
#     bootstrapped_coefs.append(clf_boot.coef_[0])
# bootstrapped_coefs = np.array(bootstrapped_coefs)
# standard_errors = np.std(bootstrapped_coefs, axis=0)
# # Plot odds ratios and confidence intervals
# plt.figure(figsize=(10, 6))
# plt.errorbar(x=(2-odds_ratios), y=X_train.columns, xerr=standard_errors, fmt='o')
# plt.axvline(x=1, color='gray', linestyle='--')
# plt.xlabel('Odds Ratio')
# plt.ylabel('Feature')
# plt.title('Odds Ratio Plot for Logistic Regression')
# plt.show()
#
#
# y_pred_proba = clf.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# #create ROC curve
# plt.plot(fpr,tpr,label="AUC="+str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()
# print("done")
