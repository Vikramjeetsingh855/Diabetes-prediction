#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# In[69]:


dataset=pd.read_csv("diabetes.csv")


# In[198]:


dataset.head(-1)


# In[71]:


dataset.isnull().sum()


# In[72]:


dataset.info()


# In[73]:


# sc=StandardScaler()
# sc.fit(dataset[["Pregnancies"]])
# dataset["Pregnancies"]=sc.transform(dataset[["Pregnancies"]])


# In[ ]:





# In[74]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["Pregnancies"])
plt.show()
# sns.kdeplot(dataset["Glucose"])


# In[75]:


q1=np.percentile(dataset["Pregnancies"],25)
q3=np.percentile(dataset["Pregnancies"],75)
q1,q3


# In[76]:


iqr=q3-q1
iqr


# In[ ]:





# In[77]:


max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[78]:


dataset.loc[dataset["Pregnancies"]>max,"Pregnancies"]=max


# In[ ]:





# In[79]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["Pregnancies"])
plt.show()


# In[80]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["Glucose"])
plt.show()


# In[81]:


q1=np.percentile(dataset["Glucose"],25)
q3=np.percentile(dataset["Glucose"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[82]:


dataset.loc[dataset["Glucose"]>max,"Glucose"]=max


# In[83]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["Glucose"])
plt.show()


# In[84]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["BloodPressure"])
plt.show()


# In[85]:


q1=np.percentile(dataset["BloodPressure"],25)
q3=np.percentile(dataset["BloodPressure"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[86]:


dataset.loc[dataset["BloodPressure"]>max,"BloodPressure"]=max
dataset.loc[dataset["BloodPressure"]<min,"BloodPressure"]=min


# In[87]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["BloodPressure"])
plt.show()


# In[88]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["SkinThickness"])
plt.show()


# In[89]:


q1=np.percentile(dataset["SkinThickness"],25)
q3=np.percentile(dataset["SkinThickness"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[90]:


dataset.loc[dataset["SkinThickness"]>max,"SkinThickness"]=max
dataset.loc[dataset["SkinThickness"]<min,"SkinThickness"]=min


# In[91]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["SkinThickness"])
plt.show()


# In[92]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["Insulin"])
plt.show()


# In[93]:


q1=np.percentile(dataset["Insulin"],25)
q3=np.percentile(dataset["Insulin"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[94]:


dataset.loc[dataset["Insulin"]>max,"Insulin"]=max
dataset.loc[dataset["Insulin"]<min,"Insulin"]=min


# In[95]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["Insulin"])
plt.show()


# In[96]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["BMI"])
plt.show()


# In[97]:


q1=np.percentile(dataset["BMI"],25)
q3=np.percentile(dataset["BMI"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[98]:


dataset.loc[dataset["BMI"]>max,"BMI"]=max
dataset.loc[dataset["BMI"]<min,"BMI"]=min


# In[99]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["BMI"])
plt.show()


# In[100]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["DiabetesPedigreeFunction"])
plt.show()


# In[101]:


q1=np.percentile(dataset["DiabetesPedigreeFunction"],25)
q3=np.percentile(dataset["DiabetesPedigreeFunction"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[102]:


dataset.loc[dataset["DiabetesPedigreeFunction"]>max,"DiabetesPedigreeFunction"]=max
dataset.loc[dataset["DiabetesPedigreeFunction"]<min,"DiabetesPedigreeFunction"]=min


# In[103]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["DiabetesPedigreeFunction"])
plt.show()


# In[104]:


plt.figure(figsize=[10,4])
sns.boxplot(dataset["Age"])
plt.show()


# In[105]:


q1=np.percentile(dataset["Age"],25)
q3=np.percentile(dataset["Age"],75)
q1,q3

iqr=q3-q1
iqr

max=q3+(1.5*iqr)
min=q1-(1.5*iqr)
max,min


# In[106]:


dataset.loc[dataset["Age"]>max,"Age"]=max
dataset.loc[dataset["Age"]<min,"Age"]=min


# In[107]:


plt.figure(figsize=[10,4])
sns.kdeplot(dataset["Age"])
plt.show()


# In[108]:


# sc=StandardScaler()
# sc.fit(dataset[["Pregnancies"]])
# dataset["Pregnancies"]=sc.transform(dataset[["Pregnancies"]])

# sc=StandardScaler()
# sc.fit(dataset[["Glucose"]])
# dataset["Glucose"]=sc.transform(dataset[["Glucose"]])

# sc=StandardScaler()
# sc.fit(dataset[["BloodPressure"]])
# dataset["BloodPressure"]=sc.transform(dataset[["BloodPressure"]])

# sc=StandardScaler()
# sc.fit(dataset[["SkinThickness"]])
# dataset["SkinThickness"]=sc.transform(dataset[["SkinThickness"]])

# sc=StandardScaler()
# sc.fit(dataset[["Insulin"]])
# dataset["Insulin"]=sc.transform(dataset[["Insulin"]])

# sc=StandardScaler()
# sc.fit(dataset[["DiabetesPedigreeFunction"]])
# dataset["DiabetesPedigreeFunction"]=sc.transform(dataset[["DiabetesPedigreeFunction"]])

# sc=StandardScaler()
# sc.fit(dataset[["Age"]])
# dataset["Age"]=sc.transform(dataset[["Age"]])


# In[ ]:





# In[109]:


dataset.head(3)


# In[110]:


x=dataset.iloc[:,:-1]
y=dataset["Outcome"]


# In[111]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[112]:


dt=DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)


# In[113]:


dt.score(x_test,y_test)*100, dt.score(x_train,y_train)*100


# In[114]:


dt.predict([[0,137,40,35,168,43.1,2.288,33]])


# In[ ]:





# In[136]:


new1=pd.read_csv("diabetes.csv")


# In[146]:


new1.head(2)


# In[147]:


new=new1.drop(columns="Outcome")


# In[150]:


new=new.head(1)
new


# In[151]:


dt.predict(new)


# In[ ]:





# In[174]:


from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder


# In[153]:


new_x=new1.drop(columns=["Outcome"])
new_x.head(2)


# In[189]:


new_y=new1["Outcome"]


# In[190]:


new_x.columns


# In[191]:


preprocessing_steps=[("scaling",StandardScaler())]


# In[192]:


classifier=DecisionTreeClassifier(max_depth=3)


# In[193]:


pipeline=Pipeline(steps=preprocessing_steps + [("classifier",classifier)])


# In[194]:


pipeline.fit(new_x,new_y)


# In[195]:


import pickle


# In[196]:


file=open("mod1.txt","wb")
pickle.dump(pipeline,file)
file.close()


# In[197]:


pipeline.predict(new_x)


# In[ ]:





# In[ ]:





# In[115]:


rf=RandomForestClassifier(criterion='log_loss', max_depth=7)
rf.fit(x_train,y_train)


# In[116]:


rf.score(x_test,y_test)*100, rf.score(x_train,y_train)*100


# In[117]:


DT={"criterion":["gini", "entropy", "log_loss"],"splitter":["best", "random"],"max_depth":[i for i in range(1,11)]}


# In[118]:


gd=GridSearchCV(DecisionTreeClassifier(),DT)
gd.fit(x_train,y_train)


# In[119]:


gd.best_estimator_


# In[120]:


RF={"criterion":["gini", "entropy", "log_loss"],"n_estimators":[i for i in range(1,20)],"max_depth":[i for i in range(1,15)]}


# In[121]:


gd=GridSearchCV(RandomForestClassifier(),RF)
gd.fit(x_train,y_train)


# In[122]:


gd.best_estimator_


# In[123]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[124]:


lr.score(x_train,y_train)*100, lr.score(x_test,y_test)*100


# In[125]:


kn=KNeighborsClassifier(n_neighbors=11)
kn.fit(x_train,y_train)


# In[126]:


kn.score(x_train,y_train)*100, kn.score(x_test,y_test)*100


# In[127]:


KN={"weights":['uniform', 'distance'],"algorithm":['auto', 'ball_tree', 'kd_tree', 'brute'],"n_neighbors":[i for i in range(1,15)]}


# In[128]:


gd=GridSearchCV(KNeighborsClassifier(),KN)
gd.fit(x_train,y_train)


# In[129]:


gd.best_estimator_


# In[ ]:





# In[ ]:




