import statsmodels.api as sm
import seaborn as sns
import numpy as np
from numpy import trunc
import pandas as pd
from numpy import trunc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt


# Readig CSV file
df = pd.read_csv("D:\\DawHacks\\Drought prediction-20240315T214358Z-002\\Drought prediction\\train_timeserie\\train_timeseries.csv")
# We can't make this work for a common path
# write your path for the data file

# Truncate the DataFrame to half of its size
small_size = len(df) // 2
#small_size = len(df) 
trunc_df = df.iloc[-small_size:]

# Write the truncated DataFrame to a new CSV file
small_training_dataset = 'small_training_dataset.csv'
trunc_df.to_csv(small_training_dataset, index=False)

# checking whether the code worked
#print(trunc_df.head())
#print(trunc_df.tail())

input_file = 'small_training_dataset.csv'
df = pd.read_csv(input_file)

#print(trunc_df.head())

#print(df.columns)

# dropping all the rows with missing values
df = df.dropna(subset=['score'])

y = df['score']
X = df.drop(['score', 'date'], axis=1)  # Droping 'score' adn'date' from independent variables

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
#print(model.summary())

# Evaliating the distribution of data in respect to drought level
columns_to_access = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET',
       'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX',
       'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN',
       'WS50M_RANGE']

score = 'score'

##plt.figure(figsize=(10, 6))
###sns.kdeplot(df[score], shade=True, color='green')
##sns.kdeplot(df[score], fill=True, color='green')
##plt.title(f'Distribution of {score}')
##plt.xlabel(score)
##plt.ylabel('Density')
##plt.show()
##
##plt.figure(figsize=(10, 6))
##plt.hist(df[score], bins=30, color='blue', alpha=0.7)
##plt.title(f'Distribution of {score}')
##plt.xlabel(score)
##plt.ylabel('Frequency')
##plt.show()


# splitting the data into the right categories
X = df[columns_to_access]  # Features
y = df['score']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# creating the model
#model = LinearRegression()
#model = linear_model.BayesianRidge()
model = linear_model.LogisticRegression()
# training the model
model.fit(X_train, y_train)

# making predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualizing the results

for i in columns_to_access:
    plt.scatter(X_test[i], y_test, color='black', label='PRECTOT values')
    plt.scatter(X_test[i], y_pred, color='blue', label='Predicted values')
    plt.xlabel(str(i))
    plt.ylabel('score')
    plt.show()





