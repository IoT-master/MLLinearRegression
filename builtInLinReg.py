from sklearn import linear_model
import pandas as pd

reg = linear_model.LinearRegression()

df = pd.read_csv('X_y_data_LinReg_XV.csv')
df.head()
print(df.describe())
X = df[list(df.columns)[:-1]]
print(X.head())
Y = df[list(df.columns)[-1]]
print(Y.head())
reg.fit(X, Y)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(5))