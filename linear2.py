#https://www.youtube.com/watch?v=jj72SdnTMfE
#https://www.youtube.com/watch?v=Co4gebvAsq8
import pandas, scipy
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Read data csv
data = pandas.read_csv('data2.csv', index_col=None)

# Define colummns as arrays
X = data.iloc[:,2:].values
Y = data.iloc[:,0].values

# plots
fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')

# data for plotting
volt = X[:,0]
ph = X[:,1]
conc = Y

# scatter plot

# surface plot just for fun - delete
triang = mtri.Triangulation(volt, ph)
ax1.plot_trisurf(triang, conc, cmap='jet')
ax1.scatter(volt,ph,conc, marker = '.', s=10, c='black',alpha=0.5)
ax1.view_init(elev=60, azim=-45)

# ax1.scatter(volt,ph,conc)
ax1.set_xlabel('voltage')
ax1.set_ylabel('ph')
ax1.set_zlabel('conc')

# calculate regression
# split into training set and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X,Y)

# find linear coeffcients
intercept = regr.intercept_
coefs = regr.coef_
print('Intercept: \n', intercept)
print('Coefficients: \n', coefs)

# conc = a1*volt + a2* ph + intercept
volt_scale = coefs[0]
ph_scale = coefs[1]

# prediction with sklearn
New_volt = 0.55
New_ph = 4.46
print('Predicted concentration', regr.predict([[New_volt, New_ph]]))

# summary table
#X = sm.add_constant(X)
#model = sm.OLS(Y, X).fit()
#predictions = model.predict(X)

#print_model = model.summary()

# test accuracy
y_pred = regr.predict(X_test)
print('Mean absolute error: %.2f' % mean_absolute_error(Y_test,y_pred))

# code for surface plot of the model
# surface plot domain
ax2 = fig.add_subplot(122, projection='3d')
x_range = y_range = np.arange(0,10,0.05)
x, y = np.meshgrid(x_range,y_range)
z = (x*volt_scale + y*ph_scale + intercept)
ax2.set_zlim(0,0.02)
ax2.set_xlabel('voltage')
ax2.set_ylabel('ph')
ax2.set_zlabel('conc')
surf = ax2.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

# put all plots onto same figure
plt.show()

# error to coefs
# normal regression
