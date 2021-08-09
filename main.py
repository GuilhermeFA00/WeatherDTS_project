#Reading csv file
import pandas as pd
data = pd.read_csv(r"C:\Users\Usuario\Desktop\I.A. lrn_path\DataScy_Learning\WeatherDTS_project\WeatherDataM.csv")

#Pre-processing

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

weather_features = ['Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)']

#"a" for input columns
a = data[weather_features]

#"b" for output columns
b = data.Humidity

#plot test
#Let's see which feature together with humidity have a good linear regression relationship
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(a['Temperature (C)'],b)
plt.subplot(2,2,2)
plt.scatter(a['Wind Speed (km/h)'],b)
plt.subplot(2,2,3)
plt.scatter(a['Pressure (millibars)'],b)
#plt.show()

#Humidity against Pressure forms no linear relationship

#Let's delete it from the model

a = a.drop("Pressure (millibars)", 1)

#Now a 3D plot
from mpl_toolkits.mplot3d import axes3d

img = plt.figure()
ax = img.add_subplot(111, projection='3d')

x1 = a["Temperature (C)"]
x2 = a["Wind Speed (km/h)"]

ax.scatter(x1, x2, b, c='r', marker='o')

ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Wind Speed (km/h)')
ax.set_zlabel('Humidity')

#plt.show()

#Implement our Multiple Linear Regression Model

from sklearn.linear_model import LinearRegression

lr_Model = LinearRegression()

#ŷ = θ₀– θ₁𝑥¹- θ₂𝑥²
lr_Model.fit(a, b)

#Use our Multiple Linear Regression Model to make predictions

b_pred = lr_Model.predict([[4, 11]])
print(b_pred)