import joblib
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping

tr = ImageDataGenerator()
traindata = tr.flow_from_directory(directory="E:/_USC/my/dogcat/train/",target_size=(224,224),shuffle=False)
val = ImageDataGenerator()
valdata = val.flow_from_directory(directory="E:/_USC/my/dogcat/validation/", target_size=(224,224),shuffle=False)

vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

##remove the last layer of the VGG16 model which is made to predict 1000 classes.
##add softmax dense layer to predict cat vs dog output
X= vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = keras.Model(inputs = vggmodel.input, outputs = predictions)
#SGD = (Stochastic Gradient Descent)
#use categorical_crossentropy, model op is categorical.
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_final.summary()

checkpoint = ModelCheckpoint("tl_vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto')
###model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 100, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
hist = model_final.fit(x= traindata, steps_per_epoch= 2, epochs= 100, validation_data= valdata, validation_steps=1,
                       callbacks=[checkpoint,early])

model_final.save("transferLearning_dogcatvgg16_1.h5")

joblib.dump(model_final,"tl_dogcatModelFinal.joblib")

