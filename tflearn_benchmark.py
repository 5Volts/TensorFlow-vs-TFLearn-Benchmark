from tflearn import fully_connected,input_data,regression,DNN
from tflearn.datasets import mnist
import time

trainX,trainY,testX,testY = mnist.load_data(one_hot=True)

model = input_data([None,784])
model = fully_connected(model,200,activation='relu')
model = fully_connected(model,10,activation='softmax')
model = regression(model,optimizer='adam',loss='categorical_crossentropy')
model = DNN(model)

start = time.time()
model.fit(trainX,trainY,n_epoch=5,validation_set=(testX,testY),show_metric=True)
print("Total time taken:",time.time()-start,'seconds')
