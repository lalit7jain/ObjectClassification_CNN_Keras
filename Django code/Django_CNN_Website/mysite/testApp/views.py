from django.shortcuts import render
from django.conf import settings
import shutil
from django.views.generic import TemplateView
from django.views.generic import CreateView
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
from fractions import gcd
from keras.utils import np_utils
import shutil
import time
from shutil import copyfile
from time import sleep
import urllib.request as urllib2
from PIL import Image
from django.core.files.storage import FileSystemStorage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from django.core.cache import cache
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.template import RequestContext
import tensorflow as tf

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
batch_size = math.gcd(trainSize, valiSize)
epochs = 1
#train Model
#preTrainModel = 0
#nameClass = None
#urlImage = []
def getImage(name, urlImage):
    
    for y in range(len(numImage)):
        service = build("customsearch", "v1",
               developerKey="AIzaSyBaSIN22OfPg9Fdf5SbCmPlwG9nK-jpvl4")

        res = service.cse().list(
            q=name,
            cx='000946463977679166157:8srxwerixf4',
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
    global preTrainModel
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
    model.save_weights(cwd+'/Models/PreTrained_finetuning_30epochs_vgg.h5')
    
    # Don't fuck with this below line.
    # else will fuck you.
    
    global graph
    graph = tf.get_default_graph()
            
    preTrainModel = model
    print(preTrainModel)
    print("*"*50)
    print(preTrainModel.input_shape)
    print("*"*50)
    print(preTrainModel.output_shape)
    print("Model Trained")
    
    
    
def beginProcessing(trainPath, valiPath):
    print("Start Processing")
    train_labels, nameOfClasses = fcount(trainPath)
    validation_labels, nameOfClasses = fcount(valiPath)
    nameOfClasses.sort()
    numClass = train_labels
    
    global nameClass
    nameClass = nameOfClasses
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
        train = 'Your Model is Successfully Trained'
        urlImage = []
        #getImage(name, urlImage)
        if len(urlImage) == 0:
            resultPath = cwd+"/Data/Train/"+name+"/"
            funCheckDir(resultPath)
            downloadImage(urlImage, resultPath, name)
            
        #adding the values in a context variable 
        context = {
            'name': name,
            'train' : train
        }
        
        template = loader.get_template('showData.html')
        
        #returing the template 
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true 
        #returing the form template 
        template = loader.get_template('index.html')
        return HttpResponse(template.render())
    
    
# new Class
class simulateView(TemplateView):
    template_name = "simulate.html"
    
def predictResult(preTrainModel,  filePath, nameClass):
    print("*"*50)
    print(preTrainModel)
    print("*"*50)
    print(nameClass)
    print("*"*50)
    print(filePath)
    print("*"*50)
    print(preTrainModel.input_shape)
    img = load_img(filePath, False, (img_width, img_height))
    x = img_to_array(img)
    print("*"*50)
    print(x.shape)
    with graph.as_default():
        predictions = preTrainModel.predict(x.reshape((1,img_width, img_height,3)),batch_size = batch_size, verbose=0)
    print("*"*50)
    print(preTrainModel.output_shape)
    result = dict()
    i = 0
    for x in predictions.tolist()[0]:
        result[nameClass[i]] = round(x,4)
        i+=1
    global finalResult
    finalResult = result
    
@csrf_exempt
def index1(request):
    if request.method == 'POST' and request.FILES['imageName']:
        #getting values from post
        myfile = request.FILES['imageName']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        filePath = cwd + "/testApp/templates/static/"+ filename
        print(filePath)
        if preTrainModel != None:
            predictResult(preTrainModel, filePath, nameClass)
        else:
            print("Model not Pre Trained")
        #dst = cwd + "/testApp/templates/static/" + filename
        #copyfile(filePath, dst)
        template = loader.get_template('simulate.html')
        context = {
            'file': myfile,
            'fResult' : finalResult
        }
        #returing the template 
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true 
        #returing the form template 
        template = loader.get_template('index.html')
        return HttpResponse(template.render())

    
def trainStaticModel():
    print("Started Training Model")
    newmodel = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=newmodel.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))
    top_model.load_weights('staticModel/Test_bottleneck_30_epochs.h5')
    model = Model(inputs = newmodel.input, outputs = top_model(newmodel.output))
    for layer in model.layers[:15]:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.load_weights('staticModel/FinalStaticModel_30epochs_vgg.h5')
    global graph1
    graph1 = tf.get_default_graph()
    
    global staticWeight
    print("Static Model Trained")
    staticWeight = model
    print(staticWeight)
    
    
    
def staticPrediction(staticWeight, filePath):
    nameClass = ['EarthQuake', 'Hurricane', 'Tornado', 'Volcanic Erruption']
    img = load_img(filePath, False, (150, 150))
    x = img_to_array(img)
    with graph1.as_default():
        predictions2 = staticWeight.predict(x.reshape((1,150, 150,3)),batch_size = 40, verbose=0)
    result = dict()
    i = 0
    for x in predictions2.tolist()[0]:
        result[nameClass[i]] = round(x,4)
        i+=1
    #global staticResult
    print(result)
    return result   

    
####### Load the static model #########    
@csrf_exempt
def index2(request):
    if request.method == 'POST':
        #getting values from post
        val = request.POST.get('predict')
        trainStaticModel()
        #adding the values in a context variable 
        context = {
            'name': staticWeight,
        }
        
        template = loader.get_template('simulateData.html')
        
        #returing the template 
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true 
        #returing the form template 
        template = loader.get_template('index.html')
        return HttpResponse(template.render()) 

###### Predict static model ############
@csrf_exempt
def index3(request):
    print("*"*50)
    print(staticWeight)
    if request.method == 'POST' and request.FILES['staticImage']:
        #getting values from post
        myfile = request.FILES['staticImage']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        filePath = cwd + "/testApp/templates/static/"+ filename
        
        if staticWeight != None:
            result = staticPrediction(staticWeight, filePath)
        else:
            print("Static Model Not Trained")
        
        template = loader.get_template('simulateData.html')
        context = {
            'file': myfile,
            'result': result
        }
        #returing the template 
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true 
        #returing the form template 
        template = loader.get_template('index.html')
        return HttpResponse(template.render())
    
    