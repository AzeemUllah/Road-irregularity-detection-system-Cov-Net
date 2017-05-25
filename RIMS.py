from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import probas_to_classes

model=Sequential()
model.add(Convolution2D(32, 5,5, input_shape=(28,28,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

train_datagen=ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
r'F:\realworld\train',
target_size=(28,28),
classes=['potholes','road'],
batch_size=1,
class_mode='categorical',
shuffle=True)

validation_generator=test_datagen.flow_from_directory(
r'F:\realworld\validation',
target_size=(28, 28),
classes=['potholes','road'],
batch_size=1,
class_mode='categorical',
shuffle=True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
early_stopping=EarlyStopping(monitor='val_loss', patience=2)
model.fit_generator(train_generator,verbose=2, samples_per_epoch=1, nb_epoch=20, validation_data=validation_generator, callbacks=[early_stopping],nb_val_samples=7)

#print(train_generator.class_indices)

count_sucess_dog=0
for num in range(1,4):
    img_path = 'F:/realworld/test/potholes/'+str(num)+'.png'
    img = image.load_img(img_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    y_proba = model.predict(x)
    y_classes = probas_to_classes(y_proba)
    
    if y_classes==0:
        count_sucess_dog = count_sucess_dog+1

count_sucess_cat=0
for num2 in range(1,4):
    img_path = 'F:/realworld/test/road/'+str(num2)+'.png'
    img = image.load_img(img_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    y_proba = model.predict(x)
    y_classes = probas_to_classes(y_proba)
    #print(train_generator.class_indices)
    if y_classes==1:
        count_sucess_cat = count_sucess_cat+1
    


print('road: Correct Prediction:' + str(count_sucess_dog) + ' Wrong Prediction: '+ str(3-count_sucess_dog)+' potholes: Correct Prediction: ' + str(count_sucess_cat) + ' Wrong Prediction: ' + str(3-count_sucess_cat))
