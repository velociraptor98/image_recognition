import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

model = vgg16.VGG16()
img=image.load_img("cat.png",target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0) 
x = vgg16.preprocess_input(x)
predictions=model.predict(x)
predicted=vgg16.decode_predictions(predictions,top=9)
print("The top possibe mathces for the image are: ")
for imageid,name,likelihood in predicted[0]:
    print("prediction:{}-{:2f}".format(name,likelihood))