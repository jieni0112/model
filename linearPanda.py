# Import libraries and classes
import pandas
import matplotlib.pyplot as plt
import numpy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Read data csv
data = pandas.read_csv('Data\data2.csv', index_col=None)

# Print data
print(data[['Voltage', 'Concentration']])

# Define colummns as arrays
Voltage = data[['Voltage']]
Concentration = data[['Concentration']]

# Split the input data into training/testing sets
Voltage_train = Voltage[:-12]
Voltage_test = Voltage[-12:]

# Split the output data into training/testing sets
Concentration_train = Concentration[:-12]
Concentration_test = Concentration[-12:]

 # Create linear regression object
regr = linear_model.LinearRegression()

 # Train the model using the training sets
regr.fit(Voltage_train, Concentration_train)

 # Make predictions using the testing set
Concentration_pred = regr.predict(Voltage_test)

 # The coefficients
print('Coefficients: \n', regr.coef_)
 # The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Concentration_test, Concentration_pred))
 # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Concentration_test, Concentration_pred))

 # Plot outputs
plt.scatter(Voltage_test, Concentration_test,  color='black')
plt.plot(Voltage_test, Concentration_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

