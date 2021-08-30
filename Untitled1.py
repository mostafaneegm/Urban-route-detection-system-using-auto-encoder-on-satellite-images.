#!/usr/bin/env python
# coding: utf-8

# In[184]:


#GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage.filters import convolve
from skimage import data, io, filters
import skimage
from skimage.morphology import convex_hull_image, erosion
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D,ZeroPadding2D, Convolution2D, ZeroPadding2D, Conv2DTranspose,ReLU
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import vgg16,vgg19,inception_v3
from keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.models import load_model
from keras import backend
#SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[185]:


Class_Dict_Path = "G:\\School\\Year 4\\Semester 2\\GP\\Roadmap project\\DATA\\RoadMap data\\class_dict.csv"
Metadata_Path = "G:\\School\\Year 4\\Semester 2\\GP\\Roadmap project\\DATA\\RoadMap data\\metadata.csv"
Main_Direction = "G:\\School\\Year 4\\Semester 2\\GP\\Roadmap project\\DATA\\RoadMap data"


# In[186]:


Reading_Class_Dict = pd.read_csv(Class_Dict_Path)
Reading_Metadata = pd.read_csv(Metadata_Path)


# In[187]:


print(Reading_Class_Dict.head())


# In[188]:


print(Reading_Metadata.head(-1))


# In[189]:


Metadata_Train = Reading_Metadata[Reading_Metadata["split"] == "train"]
Metadata_Test = Reading_Metadata[Reading_Metadata["split"] == "test"]


# In[190]:


print(Metadata_Train.head(-1))


# In[191]:


Metadata_Train.drop("split",inplace=True,axis=1)


# In[192]:


Metadata_Test.drop("split",inplace=True,axis=1)


# In[193]:


Metadata_Test.drop("mask_path",inplace=True,axis=1)


# In[194]:


print(Metadata_Test.head(-1))


# In[195]:


print(Metadata_Train.head(-1))


# In[196]:


Metadata_Train = Metadata_Train.reset_index()
Metadata_Test = Metadata_Test.reset_index()


# In[197]:


print(Metadata_Test.head(-1))


# In[198]:


print(Metadata_Train.head(-1))


# In[199]:


Metadata_Train["sat_image_path"] = Metadata_Train["sat_image_path"].apply(lambda sat_path: os.path.join(Main_Direction,sat_path))


# In[200]:


Metadata_Train["mask_path"] = Metadata_Train["mask_path"].apply(lambda mask_path: os.path.join(Main_Direction,mask_path))


# In[201]:


print(Metadata_Train.head(-1))


# In[202]:


Metadata_Test["sat_image_path"] = Metadata_Test["sat_image_path"].apply(lambda sat_path: os.path.join(Main_Direction,sat_path))


# In[203]:


print(Metadata_Test.head(-1))


# In[204]:


Metadata_Train = Metadata_Train.sample(frac=1).reset_index(drop=True)


# In[205]:


print(Metadata_Train.head(-1))


# In[206]:


Validation_Data = Metadata_Train.sample(frac=0.1,random_state=123)


# In[207]:


print(len(Validation_Data))


# In[208]:


print(Validation_Data.head(-1))


# In[209]:


Train_Data = Metadata_Train.drop(Validation_Data.index)


# In[210]:


print(len(Train_Data))


# In[211]:


print(Train_Data.head(-1))


# In[212]:


#Class process
Class_Names = Reading_Class_Dict["name"].tolist()


# In[213]:


print(Class_Names)


# In[214]:


RGB_Values = Reading_Class_Dict[["r","g","b"]].values.tolist()


# In[215]:


print(RGB_Values)


# In[216]:


Class_Type = ['background', 'road']


# In[217]:


Class_Indices = [Class_Names.index(cls.lower()) for cls in Class_Type]


# In[218]:


print(Class_Indices)


# In[219]:


Class_RGB_Values = np.array(RGB_Values)[Class_Indices]


# In[220]:


print(Class_RGB_Values)


# In[287]:


Example_Sat_Image = cv2.cvtColor(cv2.imread(Train_Data["sat_image_path"][3]),cv2.COLOR_BGR2RGB)
Example_Mask_Image = cv2.cvtColor(cv2.imread(Train_Data["mask_path"][3]),cv2.COLOR_BGR2RGB)

figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(Example_Sat_Image)
axis[1].imshow(Example_Mask_Image)


# In[277]:


Example_Sat_Image = cv2.cvtColor(cv2.imread(Train_Data["sat_image_path"][30]),cv2.COLOR_BGR2RGB)
Example_Mask_Image = cv2.cvtColor(cv2.imread(Train_Data["mask_path"][30]),cv2.COLOR_BGR2RGB)

figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(Example_Sat_Image)
axis[1].imshow(Example_Mask_Image)


# In[278]:


Example_Sat_Image = cv2.cvtColor(cv2.imread(Train_Data["sat_image_path"][587]),cv2.COLOR_BGR2RGB)
Example_Mask_Image = cv2.cvtColor(cv2.imread(Train_Data["mask_path"][587]),cv2.COLOR_BGR2RGB)

figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(Example_Sat_Image)
axis[1].imshow(Example_Mask_Image)


# In[230]:


Example_Sat_Image = cv2.cvtColor(cv2.imread(Train_Data["sat_image_path"][7]),cv2.COLOR_BGR2RGB)
Example_Mask_Image = cv2.cvtColor(cv2.imread(Train_Data["mask_path"][7]),cv2.COLOR_BGR2RGB)

figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(Example_Sat_Image)
axis[1].imshow(Example_Mask_Image)


# In[231]:


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# In[284]:


Example_Sat_Image = cv2.cvtColor(cv2.imread(Train_Data["sat_image_path"][7]),cv2.COLOR_BGR2RGB)
Example_Mask_Image = cv2.cvtColor(cv2.imread(Train_Data["mask_path"][7]),cv2.COLOR_BGR2RGB)

Mask_One_Hot_Example = []

for color in RGB_Values:
    Equality_Value = np.equal(Example_Mask_Image,color)
    Class_Value = np.all(Equality_Value,axis = -1)
    Mask_One_Hot_Example.append(Class_Value)
    
Mask_One_Hot_Example = np.stack(Mask_One_Hot_Example,axis = -1).astype("float")


# In[233]:


print(Mask_One_Hot_Example[:,:,0])


# In[234]:


print(Example_Mask_Image[:,:,0])


# In[235]:


print(Example_Mask_Image[:,:,0].shape)


# In[236]:


print(Example_Mask_Image.shape)
print(Mask_One_Hot_Example.shape)


# In[237]:


Array_Mask_Img = np.array(Example_Mask_Image)
Array_One_Hot_Img = np.array(Mask_One_Hot_Example)


# In[238]:


print(Array_Mask_Img.shape)
print(Array_One_Hot_Img.shape)


# In[239]:


print(np.argmax(Mask_One_Hot_Example, axis = -1))


# In[240]:


print(np.argmax(Mask_One_Hot_Example, axis = -1).shape)


# In[241]:


figure,axis = plt.subplots(1,2,figsize=(10,10))

axis[0].imshow(Example_Mask_Image)
axis[1].imshow(np.argmax(Mask_One_Hot_Example, axis = -1))


# In[242]:


colour_codes_example = np.array(Class_RGB_Values)
truth_mask = colour_codes_example[np.argmax(Mask_One_Hot_Example, axis = -1).astype(int)]


# In[243]:


figure,axis = plt.subplots(1,3,figsize=(10,10))

axis[0].imshow(Example_Mask_Image)
axis[1].imshow(truth_mask)
axis[1].set_xlabel(truth_mask.shape)
axis[2].imshow(Example_Sat_Image)


# In[244]:


Splitting_Data = Metadata_Train[0:3000]


# In[245]:


print(Splitting_Data.head(-1))


# In[246]:


Sat_Image = []
Mask_Image = []

for sat_img,mask_img in zip(Splitting_Data.sat_image_path,Splitting_Data.mask_path):
    Reading_Sat = cv2.cvtColor(cv2.imread(sat_img),cv2.COLOR_BGR2RGB)
    Reading_Sat = cv2.resize(Reading_Sat,(180,180))
    Reading_Sat = Reading_Sat/255.
    
    Reading_Mask = cv2.cvtColor(cv2.imread(mask_img),cv2.COLOR_BGR2RGB)
    Reading_Mask = cv2.resize(Reading_Mask,(180,180))
    Reading_Mask = Reading_Mask/255.
    
    Sat_Image.append(Reading_Sat)
    Mask_Image.append(Reading_Mask[:,:,0])


# In[247]:


print(Sat_Image[0].shape)
print(Mask_Image[0].shape)


# In[248]:


print(Sat_Image[0].dtype)
print(Mask_Image[0].dtype)


# In[249]:


Sat_Array = np.array(Sat_Image)
Mask_Array = np.array(Mask_Image)


# In[250]:


print(Sat_Array.shape)
print(Mask_Array.shape)


# In[251]:


#Model

compile_loss = "binary_crossentropy"
compile_optimizer = "adam"
compile_metrics = ["accuracy"]
input_dim = (Sat_Array.shape[1],Sat_Array.shape[2],Sat_Array.shape[3])
output_class = 1


# In[252]:


Early_Stopper = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="./modelcheck")


# In[253]:


Encoder_G = Sequential()
Encoder_G.add(Conv2D(32, (5,5),kernel_initializer = 'he_normal'))
Encoder_G.add(ReLU())
Encoder_G.add(Conv2D(64, (5,5),kernel_initializer = 'he_normal'))
Encoder_G.add(ReLU())
Encoder_G.add(Conv2D(128, (5,5),kernel_initializer = 'he_normal'))
Encoder_G.add(ReLU())


# In[254]:


Decoder_G = Sequential()
Decoder_G.add(Conv2DTranspose(64,(5,5)))
Decoder_G.add(ReLU())
Decoder_G.add(Conv2DTranspose(32,(5,5)))
Decoder_G.add(ReLU())
Decoder_G.add(Conv2DTranspose(1,(5,5)))
Decoder_G.add(ReLU())


# In[255]:


Auto_Encoder = Sequential([Encoder_G,Decoder_G])


# In[256]:


Auto_Encoder.compile(loss=compile_loss,optimizer=compile_optimizer,metrics=compile_metrics)


# In[257]:


Auto_Encoder_Model = Auto_Encoder.fit(Sat_Array,Mask_Array,epochs=5,callbacks=[Early_Stopper,Checkpoint_Model])


# In[258]:


#Prediction

Prediction_IMG = Auto_Encoder.predict(Sat_Array[:5])


# In[259]:


print(Prediction_IMG[0].shape)


# In[260]:


prediction_img_number = 1
print("NORMAL")
plt.imshow(Sat_Array[prediction_img_number])
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG[prediction_img_number])


# In[261]:


prediction_img_number = 2
print("NORMAL")
plt.imshow(Sat_Array[prediction_img_number])
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG[prediction_img_number])


# In[262]:


prediction_img_number = 3
print("NORMAL")
plt.imshow(Sat_Array[prediction_img_number])
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG[prediction_img_number])


# In[263]:


#SPECIAL PREDICTION / THE MODEL HAS NEVER SEEN BEFORE

backend.set_image_data_format('channels_last')

Non_S_IMG = cv2.cvtColor(cv2.imread("../input/satellitegooglemapsmasks/content/drive/MyDrive/Google maps/train/images/1013.jpg"),
                        cv2.COLOR_BGR2RGB)

Resize_IMG = cv2.resize(Non_S_IMG,(180,180))
Resize_IMG = Resize_IMG/255.


# In[264]:


print(Resize_IMG.shape)


# In[265]:


Resize_IMG_Prediction = Resize_IMG.reshape(-1,Resize_IMG.shape[0],Resize_IMG.shape[1],Resize_IMG.shape[2])


# In[266]:


print(Resize_IMG_Prediction.shape)


# In[267]:


Prediction_IMG_Another = Auto_Encoder.predict(Resize_IMG_Prediction)


# In[268]:


print(Prediction_IMG_Another.shape)


# In[269]:


Prediction_IMG_Another = Prediction_IMG_Another.reshape(Prediction_IMG_Another.shape[1],
                                                        Prediction_IMG_Another.shape[2],
                                                        Prediction_IMG_Another.shape[3])


# In[270]:


print(Prediction_IMG_Another.shape)


# In[271]:


print("NORMAL")
plt.imshow(Resize_IMG)
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG_Another)


# In[272]:


backend.set_image_data_format('channels_last')

Non_S_IMG = cv2.cvtColor(cv2.imread("../input/satellitegooglemapsmasks/content/drive/MyDrive/Google maps/train/images/1022.jpg"),
                        cv2.COLOR_BGR2RGB)

Resize_IMG = cv2.resize(Non_S_IMG,(180,180))
Resize_IMG = Resize_IMG/255.

Resize_IMG_Prediction = Resize_IMG.reshape(-1,Resize_IMG.shape[0],Resize_IMG.shape[1],Resize_IMG.shape[2])

Prediction_IMG_Another = Auto_Encoder.predict(Resize_IMG_Prediction)

Prediction_IMG_Another = Prediction_IMG_Another.reshape(Prediction_IMG_Another.shape[1],
                                                        Prediction_IMG_Another.shape[2],
                                                        Prediction_IMG_Another.shape[3])


print("NORMAL")
plt.imshow(Resize_IMG)
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG_Another)


# In[273]:


backend.set_image_data_format('channels_last')

Non_S_IMG = cv2.cvtColor(cv2.imread("../input/satellitegooglemapsmasks/content/drive/MyDrive/Google maps/train/images/1024.jpg"),
                        cv2.COLOR_BGR2RGB)

Resize_IMG = cv2.resize(Non_S_IMG,(180,180))
Resize_IMG = Resize_IMG/255.

Resize_IMG_Prediction = Resize_IMG.reshape(-1,Resize_IMG.shape[0],Resize_IMG.shape[1],Resize_IMG.shape[2])

Prediction_IMG_Another = Auto_Encoder.predict(Resize_IMG_Prediction)

Prediction_IMG_Another = Prediction_IMG_Another.reshape(Prediction_IMG_Another.shape[1],
                                                        Prediction_IMG_Another.shape[2],
                                                        Prediction_IMG_Another.shape[3])


print("NORMAL")
plt.imshow(Resize_IMG)
plt.show()
print("AUTO-ENCODER OUTPUT")
plt.imshow(Prediction_IMG_Another)


# In[274]:


#Check

plt.plot(Auto_Encoder_Model.history['loss'], label = 'training_loss')
plt.plot(Auto_Encoder_Model.history['accuracy'], label = 'training_accuracy')
plt.legend()
plt.grid(True)


# In[275]:


plot_model(Auto_Encoder, to_file='AEModel.png', show_shapes=True, show_layer_names=True)


# In[ ]:




