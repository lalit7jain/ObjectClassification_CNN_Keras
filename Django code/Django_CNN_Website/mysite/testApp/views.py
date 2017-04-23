from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from googleapiclient.discovery import build
import os
import requests
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
import math
from keras.utils import np_utils
import shutil
import time
from shutil import copyfile
from time import sleep
import urllib.request as urllib2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)
    
# new Class
class AboutPageView(TemplateView):
    template_name = "about.html"
    
# new Class
class getDataView(TemplateView):
    template_name = "getData.html"

def gcd(a, b):
    if(a<b):
        a,b=b,a
    while(b!=0):
        r=b
        b=a%r
        a=r
    return a
    
#disabling csrf (cross site request forgery)
numImage = [1,11,21,31,41,51,61,71,81,91]
# for Preprocessing
img_width, img_height = 150, 150
datagen = ImageDataGenerator(rescale=1./255)
cwd = os.getcwd()
trainPath = cwd+"/Data/Train/"
valiPath = cwd+"/Data/Validation/"
trainSize = sum([len(files) for r, d, files in os.walk(trainPath)])
valiSize = sum([len(files) for r, d, files in os.walk(valiPath)])
batch_size = gcd(trainSize, valiSize)
print(batch_size)
epochs = 2
#urlImage = []
def getImage(name, urlImage):
    
    for y in range(len(numImage)):
        service = build("customsearch", "v1",
               developerKey="AIzaSyBK07xoWr9EvTD888oA7EtZGnyslPdiO84")

        res = service.cse().list(
            q=name,
            cx='005673961504823902730:9y0jxagt1qc',
            searchType = "image",
            start = numImage[y],
            num=10,
            fileType='jpg',
            safe= 'high'
        ).execute()
        if not 'items' in res:
            print('No result !!\nres is: {}'.format(res))
        else:
            for item in res['items']:
                #print('{}:\n\t{}'.format(item['title'], item['link']))
                urlImage.append(item['link'])




#function to check is directory exists
def funCheckDir(path):
    print(path)
    directory = os.path.dirname(path) # defining directory path
    if not os.path.exists(directory): # checking if directory already exists
        os.makedirs(directory) # making a directory
        

def verifyImage(testImages, path):
    for i in range(len(testImages)):
        try:        
            im = Image.open(testImages[i])
            im.verify()
        except:
            src = path + '0.jpg'
            print(src)
            des =  testImages[i]
            print(des)
            copyfile(src, des)
            print("done")

            
def fcount(path):    
    nameOfClasses = [name for name in os.listdir(path)] 
    listOfclasses = len(nameOfClasses)
    return listOfclasses, nameOfClasses

cwd = os.getcwd()
def trainBottleNeck(train_data_dir, valiPath, img_width, img_height, model_vgg, train_samples, validation_samples, numClass):
    print("Train Bottleneck")
    print('*'*50)
    print(train_samples)
    print('*'*50)
    print(validation_samples)
    train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model_vgg.predict_generator(train_generator_bottleneck, train_samples // batch_size)
    modelPath = cwd+"/Models/"
    funCheckDir(modelPath)
    np.save(open(modelPath + 'bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    time.sleep(5)
    train_data = np.load(open(modelPath + 'bottleneck_features_train.npy', 'rb'))
    print('*'*50)
    
    print(len(train_data))
    b = []
    for x in range(numClass):
        for y in range(1,71):
            b.append(x)
    myarray = np.asarray(b)
    
    #train_labels = np.array([0] * (train_samples // 3) + [1] * (train_samples // 3) + [2] * (train_samples // 3))
    train_labels = myarray
    print(train_samples)
    print(train_labels.size)
    # Convert 1-dimensional class arrays to 3-dimensional class matrices (one hot vector encoding)
    train_labels = np_utils.to_categorical(train_labels, numClass)
    validateBottleNeck(train_data_dir, valiPath, img_width, img_height, model_vgg, validation_samples, train_data, train_labels, train_samples, numClass)
    

    
    
def validateBottleNeck(train_data_dir, validation_data_dir, img_width, img_height, model_vgg, validation_samples, train_data, train_labels, train_samples, numClass):
    print("Validate Bottleneck")
    validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model_vgg.predict_generator(validation_generator_bottleneck, validation_samples // batch_size)
    np.save(open(cwd+'/Models/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    time.sleep(5)
    validation_data = np.load(open(cwd+'/Models/bottleneck_features_validation.npy', 'rb'))
    print('*'*50)
    print(len(validation_data))
    b = []
    for x in range(numClass):
        for y in range(1,31):
            b.append(x)
    myarray = np.asarray(b)
    validation_labels = myarray
    print(validation_labels.size)
    #validation_labels = np.array([0] * (validation_samples // 3) + [1] * (validation_samples // 3) + [2] * (validation_samples // 3))
    # Convert 1-dimensional class arrays to 3-dimensional class matrices (one hot vector encoding)
    validation_labels = np_utils.to_categorical(validation_labels, numClass)
    model_top = Sequential()
    model_top.add(Flatten(input_shape=train_data.shape[1:]))
    model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(numClass, activation='softmax'))
    model_top.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model_top.fit(train_data, train_labels,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))
    print("Saving Weights")
    model_top.save_weights(cwd+'/Models/Test_bottleneck_30_epochs.h5')
    model_2 = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model_2.output_shape[1:]
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_2.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(numClass, activation='softmax'))
    top_model.load_weights(cwd+'/Models/Test_bottleneck_30_epochs.h5')
    #model_vgg.add(top_model)
    model = Model(inputs = model_2.input, outputs = top_model(model_2.output))
    model.summary()
    for layer in model.layers[:15]:
        layer.trainable = False
    # compile the model with a SGD/momentum optimizer and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size
        )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size
        )
    # fine-tune the model
    print("Fine tuning")
    model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)
    model.save_weights(cwd+'/Models/Test_finetuning_30epochs_vgg.h5')
    fname = glob.glob(cwd+"/Data/Train/cat/*.jpg")
    for i in range(len(fname)):    
        img = load_img(fname[i],False, (img_width, img_height))
        x = img_to_array(img)
        prediction = model.predict(x.reshape((1,img_width, img_height,3)),batch_size=32, verbose=0)
        print(np.round(prediction))
    
    
    
def beginProcessing(trainPath, valiPath):
    print("Start Processing")
    train_labels, nameOfClasses = fcount(trainPath)
    validation_labels, nameOfClasses = fcount(valiPath)
    numClass = train_labels
    ##preprocessing
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    
    train_samples = train_labels*70
    validation_samples = validation_labels*30
    model_vgg = applications.VGG16(include_top=False, weights='imagenet')
    trainBottleNeck(trainPath, valiPath, img_width, img_height, model_vgg, train_samples, validation_samples, numClass)
    #validateBottleNeck(valiPath, img_width, img_height, model_vgg, validation_samples)
    
    
    
    
                
def downloadImage(urlImage, resultPath, name):
    #print(urlImage)
    # Directory Checking Function
    cwd = os.getcwd()
    newPath = cwd+"/Data/Validation/"+name+"/"
    print(newPath)
    funCheckDir(newPath)
    count = 0
    for x in range(len(urlImage)):
        count += 1;
        if count <= 70:
            img_data = requests.get(urlImage[x], stream=True)
            with open(resultPath +"//"+ str(x) + '.jpg', 'wb') as handler:
                img_data.raw.decode_content = True
                shutil.copyfileobj(img_data.raw, handler)
        else:
            img_data = requests.get(urlImage[x], stream=True)
            with open(newPath +"//"+ str(x) + '.jpg', 'wb') as handler:
                img_data.raw.decode_content = True
                shutil.copyfileobj(img_data.raw, handler)
    trainImage = glob.glob(resultPath + "*.jpg")
    validateImage = glob.glob(newPath + "*.jpg")
    verifyImage(trainImage, resultPath)
    verifyImage(validateImage, resultPath)
    beginProcessing(trainPath, valiPath)
            



@csrf_exempt
def index(request):
    cwd = os.getcwd()
    resultPath = cwd+"/"+"Data/"
    #print(resultPath)
    #if post request came 
    if request.method == 'POST':
        #getting values from post
        name = request.POST.get('name')
        urlImage = []
        #getImage(name, urlImage)
        if len(urlImage) == 0:
            resultPath = cwd+"/Data/Train/"+name+"/"
            funCheckDir(resultPath)
            downloadImage(urlImage, resultPath, name)
            
        #adding the values in a context variable 
        context = {
            'name': name,
        }
        
            
        template = loader.get_template('showData.html')
        
        #returing the template 
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true 
        #returing the form template 
        template = loader.get_template('index.html')
        return HttpResponse(template.render())