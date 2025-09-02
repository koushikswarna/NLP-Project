import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



yelp=pd.read_csv('yelp.csv')


#1. Exploratory Data Analysis

yelp.head()
yelp.info()
yelp.describe()

yelp['text length']=yelp['text'].apply(len)

yelp.head(1)


plt.figure(figsize=(10,4))

g=sns.FacetGrid(data=yelp, row='stars')
g.map(plt.hist,'text length')
plt.show()


sns.boxplot(x='stars', y='text length', data=yelp)
plt.show()


sns.countplot(x='stars',data=yelp)
plt.show()



df=yelp.groupby('stars').mean(numeric_only=True)


df=df.corr()

sns.heatmap(df,annot=True)
plt.show()


#2.NLP Classification

yelp_class=yelp[(yelp['stars']==1)|(yelp['stars']==5)]


yelp_class.head()

x=yelp_class['text']
y=yelp_class['stars']



from sklearn.feature_extraction.text import CountVectorizer


x=CountVectorizer().fit_transform(yelp_class['text'])

#3.TrainTestSplit

from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#4.Training a Model

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

nb.fit(x_train,y_train)

#5.Predictions and Evaluations

p=nb.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,p))
print(confusion_matrix(y_test,p))

#6.Using Text Processing


from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline=Pipeline([('c',CountVectorizer()),('t',TfidfTransformer()),('m',MultinomialNB())])

#Redo Value of X and Y

X = yelp_class['text']
y = yelp_class['stars']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

pipeline.fit(x_train,y_train)

p=pipeline.predict(x_test)


#Predictions and Evaluations Final Time


print(classification_report(y_test,p))
print(confusion_matrix(y_test,p))
