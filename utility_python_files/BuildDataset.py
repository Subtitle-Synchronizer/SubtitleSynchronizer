import os
import numpy as np
from tensorflow.keras.preprocessing import image
from os import path
import pandas as pd
import pickle
from tensorflow.keras.preprocessing import image

trainImgs = []
basepath = path.dirname(__file__)
imgDirPath = os.path.join(basepath,'spectrogramImgs')
featureDF = pd.read_csv('features.csv')
goodImgDF = featureDF[featureDF['isSubTitlePresent?']==1]

for index, row in goodImgDF.iterrows():
     # access data using column names
    imgFileName = row['FileName']
    img_path = imgDirPath + '/' + imgFileName + '.png'
    #img = image.load_img(img_path, color_mode="grayscale")
    img = image.load_img(img_path)
    img = img.resize((128,128))
    img_to_array = image.img_to_array(img)
    #img_to_array = img_to_array[:, :, 0]
    trainImgs.append(img_to_array)

f = open('images.pickle', "wb")
trainImgArray = np.array(trainImgs)
f.write(pickle.dumps(trainImgArray))
f.close()
