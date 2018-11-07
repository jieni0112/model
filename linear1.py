from pandas import DataFrame
from sklearn import linear_model
import numpy as np
import statsmodels.api as sm
import csv

with open('Model Data - Soil+Fertiliser Training Data.csv') as modelData:
    data = csv.reader(modelData, delimiter=',')
    concs = []
    volts = []
    phs = []
    for row in data:
        conc = row[2]
        volt = row[4]
        ph = row[5]
        concs.append(conc)
        volts.append(volt)
        phs.append(ph)
del concs[0:2]
del volts[0:2]
del phs[0:2]

concs = [float(i) for i in concs]
volts = [float(i) for i in volts]
phs = [float(i) for i in phs]

concs = np.asarray(concs)
volts = np.asarray(volts)
phs = np.asarray(phs)

concs = concs.reshape(-1,1)
volts = volts.reshape(-1,1)
phs = phs.reshape(-1,1)

# 2 variables for multiple regression
X = volts
Y = phs

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X,Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
New_volt = 0.55
New_ph = 4.46
print('Predicted concentration', regr.predict([[New_volt, New_ph]]))
####