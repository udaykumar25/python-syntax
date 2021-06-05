## Data preprocessing and Exploratory Analysis ##
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#import data file
df = pd.read_csv("path\\AutoInsurance.csv") #encoding='latin1', encoding=''unicode_escape', encoding='utf-8', errors='ignore'
df_new = pd.concat([X, y], axis =1)
df.columns=["abc","bcd"] #replace all columns name
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1) #drop columns
data.rename(columns= { 'v1' : 'class' , 'v2' : 'message'}, inplace= True) #rename column name
data = data.iloc[:, 1:32] # Excluding id column
data=data.loc[:,["abc"]]
data=df.loc[[2,4,10,99],['Name','HP']] # df.loc[index name,column name]
df=pd.DataFrame(data)#convert to dataframe
df1<-df.copy()

# Rearrange the order of the variables
car = car.iloc[:, [1, 0, 2, 3, 4]]

##### exploratory data analysis ######
type(df)
df.dtypes
df.describe()
df.info()
df.std()
df.var()

var2.split(" ")# out put will be list

import scipy.stats
scipy.stats.skew(df.income) #skewness-0.382 
scipy.stats.kurtosis(df.income) #kurtosis=-0.864 
plt.bar(height = wcat.AT, x = np.arange(1, 110, 1))
plt.hist(df.income) #data are not normally distributed 
plt.boxplot(wcat.AT) #boxplot
sns.boxplot(df.medv);plt.title("medv");plt.show() #no outliers
plt.scatter(x = wcat['Waist'], y = wcat['AT'], color = 'green') 
plt.plot(x = wcat['Waist'], y = wcat['AT']) #line plot

########  Data processing  ######

#####MissingValues#####
# using isnull() function   
df.isnull()
#check null values in data set
df.isnull().sum()

# define the imputer
from sklearn.impute import SimpleImputer
#for mean, median impution we need sklearn library
#mean imputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
eamcet["mean"]=pd.DataFrame(mean_imputer.fit_transform(eamcet[["CUTOFF"]]))
#median imputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy="median")
eamcet["median"]=pd.DataFrame(median_imputer.fit_transform(eamcet[["CUTOFF"]]))
#mode imputer for categrical data
mode_imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
eamcet["mode"]=pd.DataFrame(mode_imputer.fit_transform(eamcet[["TYPE"]]))

####Duplication_Typecasting#####
#Identify duplicates records in the data
duplicate = df.duplicated()
sum(duplicate)
#Removing Duplicates
data1 = df.drop_duplicates() 

#####Zero Variance#####
from sklearn.feature_selection import VarianceThreshold
#using function variancethreshould.
constant_filter = VarianceThreshold(threshold=0)
a=constant_filter.fit(df.iloc[:,2:])
a.variances_

####Outlier_Treatment#######
#import below libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#box plot for numeric data to check for outliers
sns.boxplot(df.medv);plt.title("medv");plt.show()

# Detection of outliers (find limits for RM based on IQR)
IQR = df['crim'].quantile(0.75) - df['crim'].quantile(0.25)
lower_limit1 = df['crim'].quantile(0.25) - (IQR * 1.5)
upper_limit1 = df['crim'].quantile(0.75) + (IQR * 1.5)

# Now let's replace the outliers by the maximum and minimum limit
#new column in the respective dataset
df['replaced_crim']= pd.DataFrame(np.where(df['crim'] > upper_limit1, upper_limit1, 
                                         np.where(df['crim'] < lower_limit1, lower_limit1, df['crim'])))
sns.boxplot(df.replaced_crim);plt.title('replaced_crim');plt.show()

######Standardization######
#standardization using function
def function(i):
    x=(i-i.mean())/(i.std())
    return(x)
  
#df_norm=function(df.iloc[:,0])
df_norm=function(df.iloc[:,0:7])

#custom normalization
def normal(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm1=normal(df.iloc[:,0:7])

from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
# fit and transform the data
df_new = scaler.fit_transform(df)

#####Dummy Variables#######
# Create dummy variables on categorcal columns
df_new = pd.get_dummies(df,drop_first=True)
df = pd.get_dummies(df, columns = ["3D_available", "Genre"], drop_first = True)
#converting into binary
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() #factorise the data
data["checking_balance"] = lb.fit_transform(data["checking_balance"])
#######lets us see using one hot encoding works
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df1).toarray())

# converting B to Benign and M to Malignant 
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Benign ', wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant ', wbcd['diagnosis'])
#binning
data["bin"]=pd.qcut(data["Sales"],q=3,labels=["low","medium","high"])

predictors = np.array(wbcd_n.iloc[:,:]) # Predictors 
target = np.array(wbcd['diagnosis']) # Target 
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

x, y = breast_cancer.data, breast_cancer.target
# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]
