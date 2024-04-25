#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pickle
import pandas as pd


# In[22]:


dataset=pd.read_csv("diabetes.csv")


# In[23]:


p=open("mod1.txt","rb")
pipeline=pickle.load(p)


# In[ ]:





# In[26]:


dataset.head(21)


# In[33]:


Pregnancies=(int(input("enter number of pregnancies:")))
Glucose=(float(input("enter glucose:")))
BloodPressure=(float(input("enter BloodPressure:")))
SkinThickness=(float(input("enter your skin thickness:")))
Insulin=(float(input("enter Insulin amount:")))
BMI=(float(input("enter your body BMI:")))
DiabetesPedigreeFunction=(float(input("enter DiabetesPedigreeFunction:")))
Age=(int(input("enter your age:")))

outcome=pipeline.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

print("your Diabetes report is:",outcome[0])
# if outcome[0]==[1]:
#     print("your Diabetes report is positive:")
# else:
#     print("***your Diabetes report is negative***")


# In[ ]:




