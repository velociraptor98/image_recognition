import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from pathlib import Path

#load the cifar 10 data
(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

#Normalize the input image data from 0-255 in integers to the float 0-1
X_train=X_train.astype("float32")
X_test=X_test.astype("float32")
X_train=X_train/255
X_test=X_test/255

#lables stored as 0 to 9
#convert class vectors to binary class matrices
Y_train=keras.utils.to_categorical(Y_train,10)
Y_test=keras.utils.to_categorical(Y_test,10)

#build the model
#Sequential allows addition of layers one by one in a sequential manner
model=Sequential()
#convolution layers can look for patterns in an image irregadless of where the pattern occurs in an image
#conv2D is a 2 dimensional convolution layer useful for images
model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3),activation="relu"))
#max pooling allows us to scale down the complexity of our layers by keeping the larger values
#max pooling helps us to reduce the amount of parameters in our network
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
#call model.add() to add a new layer
#relu rectified linear unit 
#relu is a common activation function used with images
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
#softmax activation function ensures that all the output values in the layer add upto 1
model.add(Dense(10,activation="softmax"))
#Compile the model 
#tells keras to make the model in memory
#allows for evaluating the accuracy of the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#get summary of the model
model.summary()

#train the model
model.fit(X_train,Y_train,batch_size=32,epochs=30,validation_data=(X_test,Y_test),shuffle=True)

#save neural network structure
model_structure=model.to_json()
f =Path("model_struct.json")
f.write_text(model_structure)
#save the weights of the neural network
model.save_weights("model_weights.h5")