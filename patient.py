#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:38:21 2023

@author: shahlizaarshad
"""


'''PRE-PROCESSING'''


import pandas as pd
import numpy as np
from datetime import date


'''Train Beneficiary data'''

df1=pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
df1.columns

# create 'Deceased' column where 1 means dead
df1['Deceased'] = np.where(df1['DOD'].isna(), 0, 1)

#convert nan in DOD to 0000-00-00
df1['DOD']=df1['DOD'].fillna('2009-12-01')

# convert 'DOB' and 'DOD' columns to datetime format
df1['DOB'] = pd.to_datetime(df1['DOB'])
df1['DOD'] = pd.to_datetime(df1['DOD'])

# subtract years from 'DOD' and 'DOB' columns to get age in years
df1['Age'] = df1['DOD'].dt.year - df1['DOB'].dt.year

#drop
df1=df1.drop(columns=['DOB', 'DOD', 'RenalDiseaseIndicator', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
       'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
       'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
       'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
       'ChronicCond_stroke', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt'])


#missing values
df1.columns
#['BeneID', 'Gender', 'Race', 'State', 'County', 'Deceased', 'Age']
df1.isna().sum()

#convert datatype of BeneID
df1['BeneID']=df1['BeneID'].astype(str)


'''Train Inpatientdata'''
df2=pd.read_csv('Train_Inpatientdata-1542865627584.csv')

# convert ClaimStartDt, ClaimEndDt from string to datetime format
df2['ClaimStartDt'] = pd.to_datetime(df2['ClaimStartDt'] , format = '%Y-%m-%d')
df2['ClaimEndDt'] = pd.to_datetime(df2['ClaimEndDt'],format = '%Y-%m-%d')

# convert AdmissionDt, DischargeDt from string to datetime format
df2['AdmissionDt'] = pd.to_datetime(df2['AdmissionDt'] , format = '%Y-%m-%d')
df2['DischargeDt'] = pd.to_datetime(df2['DischargeDt'],format = '%Y-%m-%d')

# Calculate Hospitalization_Duration = DischargeDt - AdmissionDt
df2['Hospitalization_Duration'] = ((df2['DischargeDt'] - df2['AdmissionDt']).dt.days)+1
# Calculate Claim_Period = ClaimEndDt - ClaimStartDt
df2['Claim_Period'] = ((df2['ClaimEndDt'] - df2['ClaimStartDt']).dt.days)+1

#If the number of days claimed for Inpatient treatment is more than no of 
#days hospitalized is suspicious. So, I am adding this feature column
# ExtraClaimDays = Claim_Period - Hospitalization_Duration
df2['ExtraClaimDays'] = np.where(df2['Claim_Period']>df2['Hospitalization_Duration'], df2['Claim_Period'] - df2['Hospitalization_Duration'], 0)
#it was higher in 17cases

df2=df2.drop(columns=['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'ClmAdmitDiagnosisCode',
       'DeductibleAmtPaid', 'DischargeDt', 'DiagnosisGroupCode',
       'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6'])
df2.columns


#missing values
df2.isna().sum()


#add new column for In-patient=1
df2['PatientType']=1



'''Train Outpatient data'''
df3=pd.read_csv('Train_Outpatientdata-1542865627584.csv')
df3.columns


df3=df3.drop(columns=['ClaimStartDt', 'ClaimEndDt','ClmAdmitDiagnosisCode', 'DeductibleAmtPaid',
       'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6'])
df3.columns
#'BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed','AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']


#add new column for Out-patient=0
df3['PatientType']=0



'''combining'''
#combine in and out patient ie df2 and df3
df4=pd.concat([df2,df3],ignore_index=True)
df4.columns
#missing values
df4.isna().sum()

#combine df1 and df4
df5 = df4.merge(df1, on="BeneID")
df5.columns

#missing values
df5.isna().sum()

#df5=pd.concat([df1,df4],ignore_index=True)

#add fraud df
fraud_df=pd.read_csv('Train-1542865627584.csv')

#combine df5 and fraud_df
df6 = df5.merge(fraud_df, on="Provider")



#explore ifhow many times claim period was longer than the actual hospitalization duration
df6.columns
''''BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed',
       'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
       'Hospitalization_Duration', 'Claim_Period', 'ExtraClaimDays',
       'PatientType', 'Gender', 'Race', 'State', 'County', 'Deceased', 'Age',
       'PotentialFraud'''

# Replace NaN values in 'Claim_Period' and 'Hospitalization_Duration' with 0
df6.fillna({'Claim_Period': 0, 'Hospitalization_Duration': 0}, inplace=True)

#missing values
df6.isna().sum()

discrep = df6[df6['Claim_Period'] > df6['Hospitalization_Duration']]
#conclusion 17 fraud cases got reimbursement for longer claim period than hospitalization 


df7=df6.drop(columns=['Hospitalization_Duration', 'Claim_Period', 'ExtraClaimDays',
                      'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'])  

  
df7.columns
'''BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed',
       'PatientType', 'Gender', 'Race', 'State', 'County', 'Deceased', 'Age',
       'PotentialFraud'''

#cleaned data to csv
#df7.to_csv('/Users/shahlizaarshad/Desktop/Bhawesh 6212/project/558211_data.csv')


##############################################################################






'''IMPORTANT INFORMATION
we have carried out Logistic Regression on the pre-processed data (558211_data.csv provided)
in 3 different ways: 
1) all 558211 observations
2) subset of 5000 observations
3) subset of 3000 observations
    
alter the following codes accordingly to change the subset you would like to use'''

df=df7 #n=558211
#df=df7.sample(n=5000,replace=True) #n=5000
#df=df7.sample(n=3000,replace=True)  #n=3000






##############################################################################
##loading libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score
from datetime import date
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

          
df.columns
###FURTHER PRE-PROCESSING


# drop the 'ClaimID' column
df = df.drop('ClaimID', axis=1)

#removing alpha-numeric componenets
df["Provider"] = df["Provider"].str.replace("PRV", "")
df["BeneID"] = df["BeneID"].str.replace("BENE", "")

#Converting them into integers
df["Provider"] = df["Provider"].astype(int)
df["BeneID"] = df["BeneID"].astype(int)

##dummify PotentialFraud
df=pd.get_dummies(df,columns=['PotentialFraud'],drop_first=True)

# check correlations b/w variables
corr= df.corr()
#PatientType, InscClaimAmtReimbursed, Age

#############################################################################



'''Logistic Regression Model'''


df.columns
''''BeneID', 'Provider', 'InscClaimAmtReimbursed', 'PatientType', 'Gender',
       'Race', 'State', 'County', 'Deceased', 'Age', 'PotentialFraud_Yes'''

y=df[['PotentialFraud_Yes']]
x=df.drop('PotentialFraud_Yes', axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(x_train,y_train)


logmodel.coef_ #b1
'''[[-9.33975339e-07, -9.39357839e-06,  4.22682789e-05,
         6.16304465e-05,  3.18690287e-05,  9.63494585e-05,
        -1.56311054e-03,  1.54183396e-04, -2.31724523e-06,
         1.13520860e-03]])'''

y_pred=logmodel.predict(x_test)

f1_score(y_test,y_pred)
#0.005249343832020997 with n=3000
#0.04682274247491639 ith n=5000
#0.05502370940323879 with n=558211



'''Use forward selection '''

lr = LogisticRegression()

sfs = SFS(lr, 
          k_features=(1,10), 
          forward=True, 
          scoring='f1',
          cv=10)

sfs.fit(x_train, y_train)

###what features were selected
sfs.k_feature_names_
'''('PatientType','Gender', 'Race', 'State') for n=3000'''
''''(PatientType', 'Gender', 'Race', 'State', 'County') for n=5000'''
'''('PatientType','Gender', 'Race', 'State', 'County', 'Deceased', 'Age') for n=558211'''

##transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
lr.fit(X_train_sfs, y_train)
y_pred = lr.predict(X_test_sfs)

#how well the model did based on f1_score
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)
#0.20489977728285078 for n=3000
#0.1915820029027576 for n=5000
#0.18634225807299196 for n=558211
#yes model does improve if f1 improved



'''Use backward selection '''

lr = LogisticRegression()

sfs = SFS(lr, 
          k_features=(1,10), 
          forward=False, 
          scoring='f1',
          cv=10)

sfs.fit(x_train, y_train)

###what features were selected
sfs.k_feature_names_
'''('PatientType', 'Gender', 'Race', 'State')'''


##transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
lr.fit(X_train_sfs, y_train)
y_pred = lr.predict(X_test_sfs)

#how well the model did based on f1_score
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)
#0.20489977728285078 for n=3000
#0.19158200290275762 for n=5000
#0.18634225807299196 for n=558211


'''Accuracy and Confusion Matrix'''

from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, f1_score)

accuracy_score(y_test, y_pred)
#0.6215783690823102 for n=558211
c_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),index=['Actual:0','Actual:1'],columns=['Pred:0','Pred:1'])



'''Visualizations'''
#Making bar graphs for potential fraud and non fraud variable in the dataset
counts = df["PotentialFraud_Yes"].value_counts()
counts.plot(kind="bar")
# Set the title and axis labels
plt.title("PotentialFraud")
plt.xlabel("Yes(1) / No(0)")
plt.ylabel("Count")
plt.show()

#Plotting age bins in graph for potential graphs
bins = list(range(30, 121, 5))
df['AgeBin'] = pd.cut(df['Age'], bins)
# Group the data by age bin and PotentialFraud, and count the occurrences
counts = df.groupby(['AgeBin', 'PotentialFraud_Yes'])['PotentialFraud_Yes'].count()
# Unstack the counts by PotentialFraud to create a bar graph
counts.unstack().plot(kind='bar')
# Set the title and axis labels
plt.title('Potential Fraud by Age Bin')
plt.xlabel('Age Bin')
plt.ylabel('Count')
plt.show()




###############################################################################

'''Decision Tree'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Load data from file
df = pd.read_csv('fraud_only_small.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('PotentialFraud_Yes', axis=1), df['PotentialFraud_Yes'], test_size=0.3, random_state=0)

# Train the decision tree model with max_depth=3
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
dtree.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dtree.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#1.0

# Plot the decision tree
plot_tree(dtree)
plt.savefig('decision_tree.png')


###############################################################################
'''CLUSTERING'''

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

##do kmeans clustering to find clusters in food.csv
#interpret the clusters

df2=pd.read_csv('fraud_only_small.csv')
df2.columns

# remove 
df2 = df2.drop(df2.columns[0], axis=1)
df2=df2.drop(columns=['PotentialFraud_Yes', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician'])
df2.columns
df=df2


#data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df)
scaled_df=scaler.transform(df)

##clustering: import, initialize, train, interpret
#randomly i want to go with 4 clusters
wcv=[]
silk_score=[]

for i in range (2,15):
    km=KMeans(n_clusters=i,random_state=0)           #initialize
    km.fit(scaled_df)                        #training:finding the clusters 

    wcv.append(km.inertia_)          #gives within cluster variation
    silk_score.append(silhouette_score(scaled_df,km.labels_))

plt.plot(range(2,15),wcv)
plt.xlabel('no of clusters')
plt.ylabel('wcv score')

plt.plot(range(2,15),silk_score)
plt.xlabel('no of clusters')
plt.ylabel('silk score')


##lets say we go with 4
km4=KMeans(n_clusters=4,random_state=0) #initialize
km4.fit(scaled_df) #training:finding the clusters 
df['labels']=km4.labels_

#more than 2 variables so interpret with pandas rather than scatterplot
#interpret using pandas
groupby= df.groupby('labels').mean()


#Dendrogram

#Load the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# Using the dendrogram to find the optimal number of clusters
linked = linkage(scaled_df, 'ward')#gets a n-1 *4 matrix
dendrogram(linked) #uses the matrix to get to draw the dendrogram
plt.title("Dendrogram")
plt.xlabel('Customer')
plt.ylabel('euclidean')
plt.show()

#going with 6 clusters
#there is no random component here so u dont need random state
hc=AgglomerativeClustering(n_clusters=6,linkage='ward')
hc.fit(scaled_df)
df['labels']=hc.labels_

#interpret it 
hc_groupby=df.groupby('labels').mean()







































