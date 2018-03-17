import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv('/Users/akash/Documents/dataset/sample_data.csv')

df['ARR_DELAY']=df['ARR_DELAY'].apply(lambda  x:1 if x>=5 else 0)

df = pd.concat([df,pd.get_dummies(df['UNIQUE_CARRIER'],drop_first=True,prefix="UNIQUE_CARRIER")],axis=1)
df = pd.concat([df,pd.get_dummies(df['ORIGIN'],drop_first=True,prefix="ORIGIN")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DEST'],drop_first=True,prefix="DEST")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DAY_OF_WEEK'],drop_first=True,prefix="DAY_OF_WEEK")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DEP_HOUR'],drop_first=True,prefix="DEP_HOUR")],axis=1)
df.drop(['ORIGIN','DEST','UNIQUE_CARRIER','DAY_OF_WEEK','DEP_HOUR'],axis=1,inplace=True)

print df.head()

X_train, X_test, y_train, y_test = train_test_split(df.drop('ARR_DELAY',axis=1),df['ARR_DELAY'], test_size=0.30)


#Train the model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Predicting on the Test Set
predictions = logmodel.predict(X_test)

print predictions

# with open('logmodel.pkl','wb') as fid:
#     pickle.dump(logmodel,fid,2)
#
# cat=df.drop('ARR_DELAY',axis=1)
# index_dict=dict(zip(cat.columns,range(cat.shape[1])))
# with open ('cat','wb') as fid:
#     pickle.dump(index_dict,fid,2)
