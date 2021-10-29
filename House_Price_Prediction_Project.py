#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/satishgunjal/Real-Estate-Price-Prediction-Project/blob/master/House_Price_Prediction_Project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10) # width, height in inches
from google.colab import files
import io


# #Step#2: Load the data
# * Load the data in dataframe

# In[ ]:





# In[4]:


#For google colab
uploaded = files.upload()


# In[5]:


df1 = pd.read_csv(io.StringIO(uploaded['Bengaluru_House_Data.csv'].decode('utf-8')))
df1.head()


# #Step#3: Understand the data
# * Finalize the columns to work with and drop the rest of them

# In[6]:


# Get the no of rows and columns
df1.shape


# In[7]:


#Get all the column names
df1.columns


# In[8]:


#Lets check the unique values 'area_type' column
df1.area_type.unique()


# In[9]:


#Let get the count of trianing examples for each area type
df1.area_type.value_counts()


# In[10]:


# Note everytime we make change in dataset we store it in new dataframe
df2 = df1.drop(['area_type', 'availability', 'society', 'balcony'],axis='columns')

print('Rows and columns are = ', df2.shape)
df2.head()


# #Step#4: Data Cleaning
# * Check for na values
# * Verify unique values of each column
# * Make sure values are correct (eg. 23 BHK home with only 2000 Sqrft size seems wrong)

# In[11]:


# Get the sum of all na values from dataset
df2.isna().sum()


# Since null values as comapre to total training examples(13320) is verry less we can safly drop those examples

# In[12]:


df3 = df2.dropna()
df3.isnull().sum()


# In[13]:


# Since all oor training examples containing null values are dropped lets check the shape of the dataset again
df3.shape


# In[14]:


df3['size'].unique()


# In[15]:


df4 = df3.copy()

# Using lambda function we can get the BHK numeric value
df4['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
#df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df4.bhk.unique()


# From above data we can see that there are home with upto 43 BHK's in Bangalore.. must be apolitician :)

# In[16]:


#Get the training examples with home size more than 20 BHK
df4[df4.bhk >20]


# Note above 43 BHK home area is only 2400 sqrft only. We will remove this data error later. First lets clean the 'total_sqft' column

# Now lets check the unique values in 'total_sqft' column

# In[17]:


df4.total_sqft.unique()


# Note above, there are few records with range of the area like '1133 - 1384'.
# Lets write a function to identify such values

# In[ ]:


def is_float(x):
  try:
    float(x)
  except:
    return False

  return True


# In[19]:


# Test the function
print('is this (123) float value = %s' % (is_float(123)))
print('is this (1133 - 1384) float value = %s' % (is_float('1133 - 1384')))


# In[20]:


#Lets apply this function to 'total_sqft' column

#Showing training examples where 'total_sqft' vale is not float
df4[~df4['total_sqft'].apply(is_float)].head(10) 


# * Since most the value are range of sqft, we can write afunction to get the average value from a range. 
# * There are few values like '34.46Sq. Meter' and '4125Perch' we can also try and convert those values into sqft but for now I amgoing to ignore them

# In[ ]:


def convert_range_to_sqft(x):
  try:
    tokens = x.split('-')

    if len(tokens) == 2:
      return (float(tokens[0]) + float(tokens[1]))/2
    else:
      return float(x)
  except:
    return None


# In[22]:


#Lets test the convert_range_to_sqft()
print('Return value for i/p 12345 = %s' % (convert_range_to_sqft('12345')))
print('Return value for i/p 1133 - 1384 = %s' % (convert_range_to_sqft('1133 - 1384')))
print('Return value for i/p 34.46Sq. Meter = %s' % (convert_range_to_sqft('34.46Sq. Meter')))


# In[23]:


# Lets apply this function for total_sqft column
df5 = df4.copy()

df5.total_sqft = df4.total_sqft.apply(convert_range_to_sqft)
df5.head()


# In[24]:


# Since our converion function will return null for values like 34.46Sq. Meter. Lets check for any null values in it
df5.total_sqft.isnull().sum()


# In[25]:


# Lets dro the null training sets from total_sqft
df6 = df5.dropna()
df6.total_sqft.isnull().sum()

# OR
#We can also select the not null training set using below filter
#df6 = df5[df5.total_sqft.notnull()]


# In[26]:


# Lets cross check the values of 'total_sqft'
print('total_sqft value for 30th training set in df4 = %s' % (df4.total_sqft[30]))
print('total_sqft value for 30th training set in df6 = %s' % (df6.total_sqft[30]))


# In[27]:


df7 = df6.copy()

df7['price_per_sqft'] = (df6['price'] * 100000)/df6['total_sqft']
df7.head()


# In[28]:


df7_stats = df7['price_per_sqft'].describe()
df7_stats


# In[29]:


#Trim the location values
df7.location = df7.location.apply(lambda x: x.strip())
df7.head()


# In[30]:


#Lets get the count of each location
location_stats = df7.location.value_counts(ascending=False)
location_stats


# In[31]:


#Total number unique location categories are
len(location_stats)


# We are going assign a category 'other' for every location where total datapoints are less than 10

# In[32]:


#Get total number of categories where data points are less than 10
print('Total no of locations where data points are more than 10 = %s' % (len(location_stats[location_stats > 10])))
print('Total no of locations where data points are less than 10 = %s' % (len(location_stats[location_stats <= 10])))


# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns

# In[33]:


location_stats_less_than_10 = location_stats[location_stats <= 10]
location_stats_less_than_10


# In[34]:


#Using lambda function assign the 'other' type to every element in 'location_stats_less_than_10'
df8 = df7.copy()

df8.location = df7.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x )
len(df8.location.unique())


# Since 1047 location with less than 10 data points are converted to one category 'other'
# Total no of unique location categories are = 240 +1  = 241

# In[35]:


df8.head(10)


# In[36]:


# Lets visualize the data where square fit per bedroom is less than 300
df8[(df8.total_sqft / df8.bhk) < 300]


# Note abobe we have 744 training examples where square fit per bedroom is less than 300. These are outliers, so we can remove them

# In[37]:


#Lets check current dataset shape before removing outliers
df8.shape


# In[38]:


df9 = df8[~((df8.total_sqft / df8.bhk) < 300)]
df9.shape


# In[39]:


# Get basic stats of column 'price_per_sqft'
df9.price_per_sqft.describe()


# Note: Its important to understand that price of every house is location specific. We are going to remove outliers using 'price_per_sqft' for each location

# In[40]:


# Data visualization for 'price_per_sqft' for location 'Rajaji Nagar'
# Note here its normal distribuation of data so outlier removal using stad deviation and mean works perfectly here
plt.hist(df9[df9.location == "Rajaji Nagar"].price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[41]:


#Lets check current dataset shape before removing outliers
df9.shape


# In[42]:


# Function to remove outliers using pps(price per sqft)
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(mean-std)) & (subdf.price_per_sqft<=(mean + std))] # 1 Sigma value i.e 68% of data
        df_out = pd.concat([df_out,reduced_df],ignore_index=True) # Storing data in 'df_out' dataframe
    return df_out

df10 = remove_pps_outliers(df9)
df10.shape


# In[43]:


# Data visualization for 'price_per_sqft' for location 'Rajaji Nagar' after outlier removal
plt.hist(df10[df10.location == "Rajaji Nagar"].price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[44]:


# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df10,"Rajaji Nagar")


# In[45]:


plot_scatter_chart(df10,"Hebbal")


# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary by name 'bhk_stats' with below values of 'price_per_sqft'
# 
# ```
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# ```
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

# In[46]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df11 = remove_bhk_outliers(df10)
df11.shape


# In[47]:


#Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
plot_scatter_chart(df11,"Rajaji Nagar")


# In[48]:


plot_scatter_chart(df11,"Hebbal")


# Now you can campre the scatter plots for location(Hebbal and Rajaji Nagar) for before and after outlier removal

# In[49]:


#Now lets plot the histogram and visualize the price_per_sqft data after outlier removal

matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df11.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[50]:


#Get unique bath from dataset
df11.bath.unique()


# In[51]:


#Get the training examples where no of bath are more than (no of BHK +2)
df11[df11.bath > df11.bhk + 2]


# We can remove above outliers from the datset

# In[52]:


#Lets check current dataset shape before removing outliers
df11.shape


# In[53]:


# Remove the outliers with more than (no of BHK + 2) bathrooms
df12 = df11[df11.bath < (df11.bhk + 2)]
df12.shape


# This concludes our data cleaning, lets drop unnecessary columns
# * Since we have 'bhk' feature lets drop 'size'
# * We have crerated 'price_per_sqft' for outlier detection and removal purpose, so we can also drop it. 

# In[54]:


df13 = df11.drop(['size', 'price_per_sqft'], axis='columns')
df13.head()


# In[55]:


dummies = pd.get_dummies(df13.location)
dummies.head()


# In[56]:


#To avoid dummy variable trap problem lets delete the one of the dummy variable column
dummies = dummies.drop(['other'],axis='columns')
dummies.head()


# In[57]:


#Now lets add dummies dataframe to original dataframe
df14 = pd.concat([df13,dummies],axis='columns')
df14.head()


# In[58]:


#Lets delete the location feature
df15 = df14.drop(['location'],axis='columns')
df15.head()m


# #Step#5: Build Machine Learning Model
# 

# In[60]:


#Final shape of our dataset is
df15.shape


# Now leats create X(independent variable/features) and y(dependent variables/target)

# In[63]:


X = df15.drop(['price'],axis='columns')
X.head()


# In[62]:


y = df15.price
y.head()


# In[68]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)
print('X_train shape = ',X_train.shape)
print('X_test shape = ',X_test.shape)
print('y_train shape = ',y_train.shape)
print('y_test shape = ',y_test.shape)


# In[85]:


from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[70]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# ShuffleSplit is used to randomize the each fold
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv = cv)


# In[77]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

def find_best_model_using_gridsearchcv(X,y):
  algos ={
      'linear_regression':{
          'model':LinearRegression(),
          'params': {
              'normalize':[True,False]
          }
      },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
  }

  scores= []
  cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

  for algo_name, config in algos.items():
      gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
      gs.fit(X,y)
      scores.append({
          'model': algo_name,
          'best_score': gs.best_score_,
          'best_params': gs.best_params_
      })

  return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# **Based on above results we can say that LinearRegression gives the best score. Hence we will use that**.

# In[ ]:


def predict_price(location, sqft, bath, bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[86]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[87]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[88]:


predict_price('Indira Nagar',1000, 2, 2)


# In[89]:


predict_price('Indira Nagar',1000, 3, 3)


# Step#7: Export the model to Pickle file

# In[ ]:


import pickle

with open('Real_Estate_Price_Prediction_Project.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[94]:


#Since we are using the Google colab, pickle will will be saved at current directory of Google cloud machine
import os

os.listdir('.')


# In[ ]:


#Lets download it
from google.colab import files

files.download('Real_Estate_Price_Prediction_Project.pickle')


# Step#8: Export any other important info
# * since weare using One Hot Encoding for location column we need the final list of all the columns in our feature set

# In[ ]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[97]:


os.listdir('.')


# In[ ]:


files.download('columns.json')

