#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[2]:


#Import Online Retail Data containing transactions from 01/12/2010 and 09/12/2011
Rtl_data = pd.read_csv('sales_data.csv', encoding = 'unicode_escape')
Rtl_data.head(20)


# In[3]:


Rtl_data.shape


# In[4]:


Rtl_data.describe()


# In[5]:


Rtl_data.dtypes


# In[6]:


Rtl_data.info()


# In[7]:


Rtl_data.drop('Unnamed: 40', axis=1, inplace=True)
Rtl_data.drop('Unnamed: 41', axis=1, inplace=True)


# In[8]:


### Changing the data type of variable 'ORDERDATE' from object to datetime
Rtl_data['FIRST_ORDER_DATE'] = pd.to_datetime(Rtl_data['FIRST_ORDER_DATE'])


# In[9]:


Rtl_data['LATEST_ORDER_DATE'] = pd.to_datetime(Rtl_data['LATEST_ORDER_DATE'])


# # RMF Modelling

# In[10]:


Rtl_data.rename(columns={'DAYSSINCELASTORDER': 'Recency', 
                         'TOTAL_ORDERS': 'Frequency', 
                         'REVENUE': 'Monetary'}, inplace=True)

Rtl_data.reset_index().head()


# In[11]:


RFMScores=Rtl_data[['CustomerID','Recency','Frequency','Monetary']]
RFMScores.head()


# In[12]:


#Descriptive Statistics (Recency)
RFMScores.Recency.describe()


# In[13]:


#Recency distribution plot
import seaborn as sns
x = RFMScores['Recency']

ax = sns.distplot(x)


# In[14]:


#Descriptive Statistics (Frequency)
RFMScores.Frequency.describe()


# In[15]:


#Frequency distribution plot, taking observations which have frequency less than 1000
import seaborn as sns
x = RFMScores.query('Frequency < 100')['Frequency']

ax = sns.distplot(x)


# In[16]:


#Descriptive Statistics (Monetary)
RFMScores.Monetary.describe()


# In[18]:


#Monateray distribution plot, taking observations which have monetary value less than 10000
import seaborn as sns
x = RFMScores.query('Monetary < 1000')['Monetary']

ax = sns.distplot(x)


# In[19]:


#Split into four segments using quantiles
quantiles = RFMScores.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[20]:


quantiles


# In[21]:


#Functions to create R, F and M segments
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[22]:


#Calculate Add R, F and M segment value columns in the existing dataset to show R, F and M segment values
RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
RFMScores.head()


# In[23]:


#Calculate and Add RFMGroup value column showing combined concatenated score of RFM
RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)

#Calculate and Add RFMScore value column showing total sum of RFMGroup values
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
RFMScores.head()


# In[24]:


#Assign Loyalty Level to each customer
Loyalty_Level = ['champions','Potential customers','need attention' ]
Score_cuts = pd.qcut(RFMScores.RFMScore, q = 3, labels = Loyalty_Level)
RFMScores['RFM_Loyalty_Level'] = Score_cuts.values
RFMScores.reset_index().head()


# In[40]:


#Validate the data for RFMGroup = 111
RFMScores[RFMScores['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(10)


# # K-Means Clustering

# In[26]:


#Handle negative and zero values so as to handle infinite numbers during log transformation
def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
#Apply handle_neg_n_zero function to Recency and Monetary columns 
RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]

#Perform Log transformation to bring data into normal or near normal distribution
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)


# In[27]:


#Data distribution after data normalization for Recency
Recency_Plot = Log_Tfd_Data['Recency']
ax = sns.distplot(Recency_Plot)


# In[28]:


#Data distribution after data normalization for Frequency
Frequency_Plot = Log_Tfd_Data.query('Frequency < 100')['Frequency']
ax = sns.distplot(Frequency_Plot)


# In[31]:


#Data distribution after data normalization for Monetary
Monetary_Plot = Log_Tfd_Data.query('Monetary < 1000')['Monetary']
ax = sns.distplot(Monetary_Plot)


# In[32]:


from sklearn.preprocessing import StandardScaler

#Bring the data on same scale
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

#Transform it back to dataframe
Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScores.index, columns = Log_Tfd_Data.columns)


# In[33]:


from sklearn.cluster import KMeans

sum_of_sq_dist = {}
for k in range(1,15):
    km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
    km = km.fit(Scaled_Data)
    sum_of_sq_dist[k] = km.inertia_
    
#Plot the graph for the sum of square distance values and Number of Clusters
sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Sum of Square Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[34]:


#Perform K-Mean Clustering or build the K-Means clustering model
KMean_clust = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 1000)
KMean_clust.fit(Scaled_Data)

#Find the clusters for the observation given in the dataset
RFMScores['Cluster'] = KMean_clust.labels_
RFMScores.head()


# In[35]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))

##Scatter Plot Frequency Vs Recency
Colors = ["red", "green", "blue"]
RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
ax = RFMScores.plot(    
    kind="scatter", 
    x="Recency", y="Frequency",
    figsize=(10,8),
    c = RFMScores['Color']
)


# In[36]:


RFMScores.head()


# In[ ]:




