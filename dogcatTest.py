import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path


img = image.load_img("E:/_USC/my/dogcat/test1/test/8.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

saved_model = load_model("E:/_USC/my/dogcat/transferLearning_model_dogcat_vgg16_2.h5")
#'''
output = saved_model.predict(img)   ##returns probability for eah class in array format
print(output,"\n")
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')








'''
testpath= Path("E:/_USC/my/dogcat/test1/test/")
y_true=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
oparray= []
for i in testpath.glob("*.jpg"):
    img= image.load_img(i,target_size=(224,224))
    #img = np.expand_dims(img, axis=0)
    op=saved_model.predict(img)
    if op[0][0] > op[0][1]:
        oparray.append(0) #cat
    else:
        oparray.append(1) #dog

count=0
for j in range(0,np.length(oparray[0])):
    if oparray[0][j] == y_true[j] :
        count=count+1
        
print("test accuracy= ", count*100/10," (for 10 images)")
'''
