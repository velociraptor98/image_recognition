from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

#labels
labels=[
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]
#restore model structure and model weights
f=Path("model_struct.json")
model_struct=f.read_text()
model=model_from_json(model_struct)
model.load_weights("model_weights.h5")
#convert input image to 32*32 as the model was trained for that resolution
img=image.load_img("car.png",target_size=(32,32))
#convert image to numpy array
image_test=image.img_to_array(img)
list_images=np.expand_dims(image_test,axis=0)
results=model.predict(list_images)
res=results[0]
label_res=int(np.argmax(res))
likelyhood=res[label_res]
class_label=labels[label_res]
print("this is a image of a ",class_label," with likelihood of",likelyhood)