import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# read the dataset
data = pd.read_csv('Iris.csv')
print(data.head())


print(data.columns)

#label encode the target variable
encode = LabelEncoder()
data['Species'] = encode.fit_transform(data['Species'])

print(data.head())

x = data.iloc[:, :4].values
Y = data.iloc[:, -1:].values


# train-test-split   
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2,random_state=0)


#Decision Tree 100% accuracy
'''
from sklearn.tree import DecisionTreeRegressor , export_graphviz
model = DecisionTreeRegressor(random_state = 0)
model.fit(x_train, Y_train)


#SEE DECISION TREE ON CHART

export_graphviz(model,out_file="model.dot")
                
with open("model.dot") as models:
    model_graph = models.read()
graphviz.Source(model_graph)

predict = model.predict(x_test)
'''
# create the object of the model
model = LogisticRegression()

model.fit(x_train, Y_train)

predict = model.predict(x_test)

print('Predicted Values on Test Data',encode.inverse_transform(predict))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predict)*100,'%')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict)
