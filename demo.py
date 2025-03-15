import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("sales_data.csv")

print(df.columns)

df.groupby("Region")["Sales_Amount"].sum().sort_values(ascending=False).plot(kind="bar")
plt.show()

df["Profit"] = (df['Unit_Cost']-df['Unit_Price']) * df['Quantity_Sold']

df.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False).plot(kind="bar")
plt.show()

df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
df['Month'] =df['Sale_Date'].dt.to_period("M")
m_s = df.groupby("Month")['Sales_Amount'].sum()

sns.lineplot(x= m_s.index.astype(str),
		y=m_s.values,color="red",marker="*")
plt.show()


X =df[["Unit_Price"]]
y = df["Sales_Amount"]


from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.3,random_state = 42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)

sns.scatterplot(x=X_test['Unit_Price'],y=y_test,label = "actual_sales")
sns.lineplot(x=X_test['Unit_Price'],y=y_pred,label="predicted sales",color="red")

plt.show()




