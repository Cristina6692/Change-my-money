import numpy as np
import pandas as pd
from PIL import Image
from cv2 import cv2
import glob
import os



def readImages(path,labels):
    '''path: string con la ruta de la carpeta general (Ej:"./Inputs/coins")'''
    '''labels: lista con las strings de las carpetas que contienen las imagenes(Ej:["1c","2e"]) '''

    imagenes=[]
    img_label=[]
    img_path=[]
    
    for label in labels:
        basedir = glob.glob(path+'/'+label+'/resized/*.jpg')
        for im in basedir:
            img_label.append(label)
            img_path.append(os.path.relpath(im))
            imagenes.append(cv2.imread(im))
    
    return imagenes,img_label,img_path


def rotarImgX(path,grados):
    '''path debe ser la ruta a la carpeta que contenga las fotos que queremos rotar 90º'''
    '''Ej: './Inputs/coins/2c' '''

    basedir = glob.glob(path+'/*.jpg')
    counter = 0
    for im in basedir:
        counter+=1
        img = cv2.imread(im)
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, grados, scale)
        img_gr= cv2.warpAffine(img, M, (h, w))
        cv2.imwrite(path+f'/img_{grados}_'+str(counter)+'.jpg', img_gr)

    return f'Has creado {counter} nuevas imagenes rotadas {grados} grados'


