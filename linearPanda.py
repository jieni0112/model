# # Import libraries and classes
#
# import numpy
# import matplotlib.pyplot as plot
# import pandas
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
#
# # Import dataset
# # Integer selection with pandas: data.iloc[<row selection>, <column selection>]
# data = pandas.read_csv('Data\modelData.csv')
#
# Concentration = data.iloc[:, 0]
# Ratio = data.iloc[:, 1]
# Voltage = data.iloc[:, 2]
# pH = data.iloc[:, 3]
#
# print(type(data.iloc[:, 0]))
#
# # x = dataset.iloc[:, :-1].values
# # y = dataset.iloc[:, 1].values
#


import pandas

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


data = pandas.read_csv('Data\data2.csv', index_col=None)

print('Concentration, Ratio')
print(data[['Voltage']])



regr = linear_model.LinearRegression()

regr.fit(data[['Voltage']], data[['Ratio']])

plt.scatter(data[['Voltage']], data[['Ratio']],  color='black')

plt.xticks(())
plt.yticks(())

plt.show()

## define the data/predictors as the pre-set feature names
#inputs = pandas.DataFrame(data.data, columns=["pH"])

#target = pandas.DataFrame(data.data, columns=["Concentration"])




#  # Load the diabetes dataset
# diabetes = datasets.load_diabetes()
#
#
#  # Use only one feature
# diabetes_X = diabetes.data[:, np.newaxis, 2]
#
#  # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]
#
#  # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]
#
#  # Create linear regression object
# regr = linear_model.LinearRegression()
#
#  # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)
#
#  # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)
#
#  # The coefficients
# print('Coefficients: \n', regr.coef_)
#  # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
#  # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
#
#  # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()