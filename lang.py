import pickle
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
df = pd.read_csv('data.csv')
print(df.head())
X = df['Text']
y = df['Language']

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]',' ', text)
    text = text.lower()
    data_list.append(text)

X = data_list

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

lang_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

lang_clf.fit(X_train, y_train)
Predictions = lang_clf.predict(X_test)
print(confusion_matrix(y_test, Predictions))


pickle.dump(lang_clf,open('modelcan.pkl','wb'))
lang_clf=pickle.load(open('modelcan.pkl','rb'))