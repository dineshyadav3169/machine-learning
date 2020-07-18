# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('kidney_disease.csv')

#Print the first 5 rows
df.head()

#Get the shape of the data (the number of rows & columns)
df.shape

#Count the empty (NaN, NAN, na) values in each column
df.isna().sum()

#Get columns of df
df.columns

#Create a list of columns to retain
columns_to_retain = ["sg", "al", "sc", "hemo",
                         "pcv", "wbcc", "rbcc", "htn", "classification"]

#Drop the columns that are not in columns_to_retain
#NOTE: in drop or dropna axis=1 means column and axis=0 means rows
for col in df.columns:
    if col not in columns_to_retain:
        df = df.drop(col, axis=1)
        
# Drop the rows with na or missing values
df = df.dropna(axis=0)

#Look at the data types to see which columns need to be transformed / encoded to a number
df.dtypes

#Transform non-numeric columns into numerical columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['classification'] = encoder.fit_transform(df['classification'])
df['htn'] = encoder.fit_transform(df['htn'])
df['pcv'] = df['pcv'].astype(int)

    
#Print / show the first 5 rows of the new cleaned data set
df.head()

#Split the data into independent'X'(the features) and dependent 'y' variables (the target)
X = df.drop(["classification"], axis=1)
y = df["classification"]

#Split the data into 80% training and 20% testing & Shuffle the data before splitting
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Build model
#  The models input shape/dimensions is the number of features/columns in the data set
#  The model will have 2 layers:
#      (i) The first with 256 neurons and the ReLu activation function & a initializer which 
#          defines the way to set the initial random weights of the Keras layers. 
#          We'll use a initializer that generates tensors with a normal distribution.
#     (ii) The other layer will have 1 neuron with the activation function 'hard_sigmoid'

import keras as k
from keras.layers import Dense
from keras.models import Sequential, load_model

model = Sequential()
model.add(Dense(256, input_dim=len(X.columns),
                    kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
model.add(Dense(1, activation="hard_sigmoid"))

#Compile the model
# Loss measuers how well the model did on training , and then tries to improve on it using the optimizer.
# The loss function we will use is binary_crossentropy for binary (2) classes.
model.compile(loss='binary_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

#Train the model
history = model.fit(X_train, y_train, 
                    epochs=2000, #The number of iterations over the entire dataset to train on
                    batch_size=X_train.shape[0]) #The number of samples per gradient update for training

#Save the model
model.save("kidney_disease.model")


#Visualize the models accuracy and loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("model accuracy & loss")
plt.ylabel("accuracy and loss")
plt.xlabel("epoch")
plt.legend(['accuracy', 'loss'], loc='upper right')
plt.savefig('accuracy_loss.png', dpi=1200)




#Loop through any and all saved models. Then get each models accuracy, loss, prediction and original values on the test data.

model = load_model('kidney_disease.model')
pred = model.predict(X_test)
#converting float numbers to int if number is >= 0.5 then it is 1 else 0
for j in range(len(pred)):
    if pred[j]>=0.5:
        pred[j]=1
    else:
        pred[j]=0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)


#Testing model on new dataset
scores = model.evaluate(X_test, y_test)
print("Scores    : loss = ", scores[0], " acc = ", scores[1])