import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
from cv2 import cv2
import glob
import os



def readImages(path,labels):
    '''path: string con la ruta de la carpeta general (Ej:"./Inputs/coins")'''
    '''labels: lista con las strings de las carpetas que contienen las imagenes(Ej:["1c","2e"]) '''

    img_array=[]
    img_label=[]
    img_path=[]
    
    for label in labels:
        basedir = glob.glob(path+'/'+label+'/*.jpg')
        for im in basedir:
            img_label.append(label)
            img_path.append(os.path.relpath(im))
            img_array.append(cv2.imread(im))
    
    return img_array,img_label,img_path


def rotAndTransfImg(img_array,grados):
    '''Input: array de imagen, grados a rotar'''
    '''Output: Devuelve el array de la imagen rotada'''
 
    img = ndimage.rotate(img_array, grados)
    img = cv2.resize(img,(70,70))
    img = np.stack(img) / 255.0

    return img



def transfImg(img_array):
    '''Transforma imagen en array, hace resize(70x70) y la convierte en apta para entrenar el modelo'''
    print (f'transforming image to 70x70')
    
    img_data = cv2.resize(img_array,(70,70))
    img_data = np.stack(img_data) / 255.0
    
    return img_data


def rotateConcat(coins_original, coins, grados):
    ''' Rota y transforma los arrays de un dataframe y los concatena al otro df con sus respectivas labels'''

    coins_gr = pd.DataFrame()
    coins_gr['image'] = coins_original['image'].apply(lambda x: rotAndTransfImg(x,grados))
    coins_gr = coins_gr.join(coins_original['label'])
    coins = pd.concat([coins,coins_gr])
    print(coins.shape)
    return coins


def videoToFrames(path):
    '''Read the video from specified path '''

    direc = glob.glob(path)
    for video in direc:
        cam = cv2.VideoCapture(video) 
        try: 
            # creating a folder named data 
            if not os.path.exists('data'): 
                os.makedirs('data') 
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data')        
        # frame 
        currentframe = 0
        while(True): 
            # reading from frame 
            print('reading image')
            ret,frame = cam.read() 
            if ret: 
                # if video is still left continue creating images 
                name = './data/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name) 
                # writing the extracted images 
                cv2.imwrite(name, frame) 
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                print('Error')
                break

