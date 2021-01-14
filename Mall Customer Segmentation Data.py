#!/usr/bin/env python
# coding: utf-8

# In[3]:


## For data manupilation
import numpy as np
import pandas as pd
# For os
import os
# remove warning
import warnings
warnings.filterwarnings("ignore")


# In[4]:


os.chdir("D:\\data analyst data\\download data\\Mall Customer Segmentation Data")
os.listdir()


# In[5]:


df = pd.read_csv('Mall_Customers.csv')
df.head() 


# In[6]:


df.isnull().sum() # checking if there is any null value 


# In[7]:


df.rename(columns= {'Annual Income (k$)' : 'Annual_Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)


# In[8]:


df.drop('CustomerID', axis = 1 ,inplace = True)


# In[9]:


df.dtypes # checking data type 


# In[10]:


df.shape #checking shape of data 


# In[11]:


df.columns.values #checking columns 


# In[12]:


# For creating colormaps
import matplotlib.pyplot as plt
import seaborn as sns
# for clustering 
from sklearn.cluster import KMeans
import time  # Measuring process time
from sklearn.preprocessing import StandardScaler #  For data processing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt


# In[13]:


np.min(df.Age), np.max(df.Age)


# In[14]:


sns.distplot(df.Age) # diagrams 


# In[15]:


sns.distplot(df.Annual_Income, hue = 'Gender')


# In[ ]:


sns.distplot(df.Spending_Score)


# In[16]:


sns.jointplot(df.Age, df.Spending_Score)


# In[17]:


sns.jointplot(df.Annual_Income, df.Age , kind="kde")


# In[18]:


sns.jointplot(df.Annual_Income, df.Age , kind="hex")


# In[19]:


sns.jointplot(df.Annual_Income, df.Age , kind="reg")


# In[20]:


cust_group = df.groupby(['Gender','Age']).sum().reset_index()
cust_group


# In[21]:


df['Annual_Income'].min()


# In[22]:


df['Annual_Income'].max()


# In[23]:


df['Annual_Income_cat'] = pd.cut(
                       df['Annual_Income'],
                       bins = 3,
                       labels= ['low', 'medium','high']
                      )
df


# In[24]:


plt.figure(1 , figsize = (10 , 10))
sns.barplot(x = 'Annual_Income_cat', y = 'Age',hue = 'Gender', data = df, )


# In[25]:


plt.figure(figsize=(15,3))
sns.barplot(data=cust_group,x='Age',y='Spending_Score',hue='Gender')


# In[26]:


sns.pairplot(data=cust_group, hue='Gender')


# In[27]:


df.drop(columns=['Annual_Income_cat'],inplace= True)  #Dropping columns not needed


# In[28]:


df


# In[29]:


from sklearn.preprocessing import LabelEncoder as le
enc=le()


# In[31]:


df['Gender']=enc.fit_transform(df['Gender'])
df.head()


# In[32]:


find_cls=[]
for i in range(1,15):
    kmean = KMeans(n_clusters=i)
    kmean.fit(df)
    find_cls.append(kmean.inertia_)


# In[33]:


find_cls


# In[34]:


fig, axs = plt.subplots(figsize=(12,5))
sns.lineplot(range(1,15),find_cls, ax=axs,marker='X')
axs.axvline(5, ls="--", c="crimson") # CRIMSON is color, ls - line style
axs.axvline(6, ls="--", c="crimson")
plt.grid() #  square lines in back ground
plt.show


# In[35]:


kmean=KMeans(n_clusters=5) # we found that best clusters are 5
kmean.fit(df)


# In[36]:


kmean.inertia_


# In[37]:


kmean.cluster_centers_


# In[38]:


clust_centers=kmean.cluster_centers_


# In[39]:


df.head()


# In[40]:


kmean.labels_


# In[41]:


df['center_cluster']=kmean.labels_


# In[42]:


df


# In[47]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))
sns.scatterplot(data = df, x = 'Age', y = 'Spending_Score', ax=ax1, hue = 'center_cluster',palette='Set1')  # good color use palette =1
sns.scatterplot(data = df, x='Annual_Income', y ='Spending_Score', ax=ax2, hue='center_cluster',palette='Set1')


# In[344]:


fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(15,5))

ax1.scatter(kmean.cluster_centers_[:,1], kmean.cluster_centers_[:,3],marker='X',color='red')      
ax2.scatter(kmean.cluster_centers_[:,2], kmean.cluster_centers_[:,3],marker='X',color='red')


# In[ ]:





# In[178]:


# Copy 'Age' column to another variable and then drop it
#     We will not use it in clustering
y = df['Age'].values
df.drop(columns = ['Age'], inplace = True)


# In[179]:


y


# 
