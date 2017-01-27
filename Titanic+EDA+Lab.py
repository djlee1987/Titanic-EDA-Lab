
# coding: utf-8

# In[1]:


"""
Created on Thu Jan 26 20:30:17 2017

@author: djlee1987 Daniel Lee
UIS - Data Science Essentials
Titanic EDA Lab
"""


# In[2]:

import pandas as pd
get_ipython().magic('pylab inline')


# In[3]:

dfMain = pd.read_csv("train.csv")


# ### conduct a quick visual inspection of data to get familiar with what's inside

# In[4]:

dfMain


# In[5]:

dfSummary = dfMain.describe() ##generate summary statistics of dfMain


# In[6]:

dfSummary


# In[7]:

# Create and format EDA df will appropriate values and results http://pandas.pydata.org/pandas-docs/stable/10min.html
dfEDA = dfSummary.loc[:,['PassengerId', 'Survived', 'Pclass']]
dfEDA['Name'] = None
dfEDA['Sex'] = None
dfEDA = pd.concat([dfEDA, dfSummary[['Age', 'SibSp', 'Parch']]], axis = 1)
dfEDA['Ticket'] = None
dfEDA = pd.concat([dfEDA, dfSummary[['Fare']]], axis = 1)
dfEDA['Cabin'] = None
dfEDA['Embarked'] = None

# Add Rows for 'contin' and 'misvals'
"""
'contin' row will identify Column/Variable as
True for continuous or False for Categorical
"""
dfEDA.loc['contin'] = False
""" 
'misvals' row will identify data in Column/Variable as 
True for Yes, there is missing Data or False for No, there isn't any missing data
"""
dfEDA.loc['misvals'] = False



# ### 'contin' row will identify Column/Variable as 
# <li> True for continuous
# <li> False for Categorical
# 
# ### 'misvals' row will identify data in Column/Variable as 
# <li> True for Yes, there is missing Data
# <li> False for No, there isn't any missing data

# In[8]:

# populate 'count' row with missing data
misCols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
for x in misCols:
    dfEDA.loc['count', x] = len(dfMain[x]) - len(dfMain[dfMain[x].isnull()]) # count = column length - no. of nulls


# In[9]:

# graph all variables
dfMain['PassengerId'].hist()
print("PassengerId's data is a continuous variable and follows a continuous uniform distribution")


# In[10]:

dfMain['Survived'].hist(bins = 5)
print("Survived's data is a categorical variable and follows a Bernoulli distribution")


# In[11]:

dfMain['Pclass'].hist()
# distribution guide at http://people.stern.nyu.edu/adamodar/New_Home_Page/StatFile/statdistns.htm
print("Pclass's data is a categorical variable and follows a non-monotonic beta distribution with negative skew")


# In[12]:

dfMain.Sex.value_counts().plot(kind='bar')
print("Sex's data is a categorical variable and follows a Bernoulli distribution")


# In[13]:

dfMain['Age'].hist(bins=80)
print("Age's data is a continuous variable and follows a gamma distribution")


# In[14]:

dfMain['SibSp'].hist(bins = 8)
print("SibSp's data is a continuous variable and follows an exponential distribution")


# In[15]:

dfMain['Parch'].hist(bins = 6)
print("Parch's data is a continuous variable and follows an exponential distribution")


# In[16]:

dfMain.Ticket.value_counts().plot(kind='bar')
print("Parch's data is a categorical variable that's relatively evently distributed across nearlly all categories")


# In[17]:

dfMain['Fare'].hist(bins=50)
print("Fare's data is a continuous variable and follows an exponential distribution")


# In[18]:

dfMain.Cabin.value_counts().plot(kind='bar')
print("Parch's data is a categorical variable that's relatively evently distributed across nearlly all categories")


# In[19]:

dfMain.Embarked.value_counts().plot(kind='bar')
print("Embarked's data is a categorical variable and with Southapmton embarkment encompassing most the distribution")


# In[20]:

# populate 'conti' row
## set all values to False and change those that are continuous to True
colnames = dfEDA.columns
dfEDA.loc['contin'] = False
conticols = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
for x in colnames:
    for y in range(0,5):
        if (x == conticols[y]):
            dfEDA.loc['contin', x] = True


# In[21]:

# populate 'misvals' row
for x in colnames:
    if len(dfMain[dfMain[x].isnull()]) > 0:
        dfEDA.loc['misvals',x] = True
    else:
        dfEDA.loc['misvals',x] = False


# In[22]:

dfEDA


# ### So we have several categories with missing values but most concering is Age, which also a continuous variable.  For the puposes of this exercise, we will use mean(age) to replace all Null values.  The we will create and populate similar df's named dfMainAdj and dfEDAAdj

# In[23]:

# replicate necessary tables, find average age and replace Null values with mean(age)
dfMainAdj = dfMain
dfEDAAdj = dfEDA
avgAge = dfMain.Age.mean()
newAgeCol = dfMainAdj.Age.fillna(value = avgAge)
dfMainAdj.Age = newAgeCol


# In[24]:

# calculate new summary statistics for age and place into dfEDAAdj
newAgeStats = newAgeCol.describe()
dfEDAAdj.Age = newAgeStats


# In[25]:

# fill 'contin' and 'misvals' rows
dfEDAAdj.loc['contin', 'Age'] = True
dfEDA.loc['misvals','Age'] = False


# In[26]:

dfEDAAdj


# In[27]:

# re-graph adjusted Age histogram and record observations
dfMain['Age'].hist(bins=80)
print("Age's distribution is compromised by the mean(age) we replaced all null values with")


# In[ ]:



