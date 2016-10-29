# 
# Author: Sunil
# credits: https://rpubs.com/dvorakt/144238
#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import json
from pprint import pprint

with open('data.json') as df:    
    data = json.load(df)

emails = list()
types = list()

for d in data:
    emails.append(d["message"])
    types.append(d["type"])

evec = CountVectorizer()
features = evec.fit_transform(emails)

le = preprocessing.LabelEncoder()
labeler = le.fit(types)
target = labeler.transform(types)

gnb = GaussianNB()
model = gnb.fit(features.todense(), target.T)

test = ["viagra meet"]
Xpredict = evec.transform(test)

pl = model.predict(Xpredict.todense())
print(le.inverse_transform(pl))

#
# useful links: 
#   http://stackoverflow.com/questions/19984957/scikit-predict-default-threshold
#   http://scikit-learn.org/stable/modules/naive_bayes.html
#   https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/
#