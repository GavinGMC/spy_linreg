import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



df = pd.read_csv('SPY.csv')

#Checking for nulls
#print("Are there any nulls in the dfset?: "+ str(df.isnull().values.any())) 


#plt.plot(df['Date'],df['Close'])
#plt.show()

#Converts string into DateTime
df['Date'] = pd.to_datetime(df['Date'])
#Converts DateTime "Date" into integer value
df['Date'] = df['Date'].dt.strftime("%Y%m%d").astype(int)


X = df['Date']
y = df['Close']

X = X.values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



plt.scatter(X_test, y_test, color='black', label='Data Points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.title('SPY Stock Price Prediction')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.legend()
plt.show()