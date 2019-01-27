#%% [markdown]
# ## Importing the libraries
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#%% [markdown]
# ## Importing the dataset
#%%
dataset = pd.read_csv('50_Startups.csv')
#%% [markdown]
# ## Spliting data into independent variables (X) and dependent variables (y)
#%%
X = pd.DataFrame(dataset.iloc[:, :-1],columns=dataset.columns[:-1])# independent 
y = pd.DataFrame(dataset.iloc[:, 4],columns=['Profit']) # dependent
#%% [markdown]
# ## Take a look at X
#%%
X.head(10)
#%% [markdown]
# ## Take a look at y
#%%
y.head(10)
#%% [markdown]
# ## Encoding categorical data (state feature)
# We use the pandas get_dummies function to one hot encode the data.
# We specify drop_first=True to drop one of the one hot encoded variables.
# We do this to avoid multicollinearity.
# In our case sklearn takes care of this silently even if we don't tell get_dummies, but to me it seems like best practice.
#%% 
X = pd.get_dummies(X, columns=['State'],drop_first=True)
#%% [markdown]
# ## Take a look at X after encoding the states
#%%
X.head(10)
#%% [markdown]
# Notice there is no column for State_California
#%% [markdown]
# ## Splitting the dataset into the Training set and Test set
# We go with a 20% test size as this is pretty standard.
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
#%% [markdown]
# The random_state=0 is simply there for reproducibility of the results.
#%% [markdown]
# ## Let's take a look at X_train
#%%
X_train.head()
#%% [markdown]
# ## and now y_train
#%%
y_train.head()
#%% [markdown]
# Take notice that X_train's index columns is the same as y_train's.
#%% [markdown]
# ## Now let's look at X_test
#%%
X_test.head()
#%% [markdown]
# ## and y_test
#%%
y_test.head()
#%% [markdown]
# Same as above, the index columns of the X_test and y_test are the same.
#%% [markdown]
# ## Here we are fitting a linear regression model to the Training set.
#%%
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#%% [markdown]
# ## Getting the r squared score .
#%%
regressor.score(X,y) * 100
#%% [markdown]
# This value is the % of the variability in profit that can be explained by the model. 94.85% in our case.
#%% [markdown]
# ## Now we're predicting the Test set results.
#%%
predictions = regressor.predict(X_test)
#%% [markdown]
# ## Let's see how we did by checking the mae (mean absolute error).
#%%
mae = mean_absolute_error(predictions,y_test)
mae
#%% [markdown]
# So, on average our model is off by $7514.29.
#%% [markdown]
# ## Let's put that into context by seeing it as a percent.
#%%
float(mae/y_test.mean() * 100)
#%% [markdown]
# So, on average our model is only off by 6.15%, not too bad.
#%% [markdown]
# In the future we could do feature elimination in order to try to remove uneeded features and make our model even better.
#%% [markdown]
# ### Note
# The 50_Startups.csv was taken from the Machine Learning A-Z™: Hands-On Python & R In Data Science course offered on Udemy
# This served as an exercise for me to delve into machine learning and data science as well as work with Jupyter.
