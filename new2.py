import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
file_path = "spam.csv"
from sklearn.naive_bayes import MultinomialNB #THIS IS the model 
from sklearn.metrics import classification_report


# dataset consist of spam and non spam emails

df = pd.read_csv(file_path)



df = df.drop(df.columns[[2,3,4]],axis =1)	
df.columns = ['category','message']


# the above 2 lines are only for cleaning the data removing unwanted columns and naming the columns




df['spam'] = df['category'].apply(lambda x: 1 if x =='spam' else 0)



x_train,x_test,y_train,y_test = train_test_split(df.message,df.spam,test_size=0.2,shuffle=False)
# print(x_train[690])

v = CountVectorizer()

x_train_cv = v.fit_transform(x_train.values)  #this 
	


x_train_np = x_train_cv.toarray()

position = np.where(x_train_np[0] != 0)

model = MultinomialNB()

model.fit(x_train_cv,y_train)
# print(y_train)
x_test_cv = v.transform(x_test)

y_pred = model.predict(x_test_cv)


# print(classification_report(y_test,y_pred))


# k = ''
email_path = "email_path.txt"
with open(email_path,'r') as file:
	k = file.read()

print('the email is :' , k)

emails = [k] #used for testing


emails_count = v.transform(emails)


# print(emails_count)
k = model.predict(emails_count)



if k[0] == 1:
	print('it is spam')
else :
	print('it is not a spam email')

 

#k[0] is assuming only one email is used if multiple emails are 
#present the array will be of 1 and 0 indicating if it is 
# spam or not




# print(x_train_cv[0][0])
# print(x_train_np[0])


# print(np.where(x_train_np[0] != 0))