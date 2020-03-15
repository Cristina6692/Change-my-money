from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def newImag(path):
    print ('transforming image from {}'.format(path))

    img=cv2.imread(path)
    img_data=cv2.resize(img,(70,70))
    
    img_data = np.stack(img_data)
    img_data = img_data / 255.0
    
    return img_data


def whoIam(path, model):

    ''' predicción si es 1€,2€, dorada, o de cobre'''
    PIC = newImag(path) # transform pic

    img=cv2.imread(path) # get the array of the original pic
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img) # original pic


    PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 70, 70, 3)
    print('transformed img shape: ', PIC.shape)
    pred2 = model.predict(PIC)[0]
    print('predicction: ', pred2)
    return "Probs -> 1€:{0:.5f} 2€:{1:.5f} AU:{2:.5f} CU:{3:.5f}".format(pred2[0],pred2[1],pred2[2],pred2[3])


def whoIamAU(path,model):
    ''' predicción que moneda dorada es'''
    PIC = newImag(path) # transform pic

    img=cv2.imread(path) # get the array of the original pic
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img) # original pic


    PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 70, 70, 3)
    print('transf img shape: ', PIC.shape)
    pred3 = model.predict(PIC)[0]
    print('Prediccion: ', pred3)
    return "Probs -> 10c:{0:.5f} 20c:{1:.5f} 50c:{2:.5f}".format(pred3[0],pred3[1],pred3[2])

def whoIamCU(path,model):
    ''' predicción que moneda de cobre es'''
    PIC = newImag(path) # transform pic

    img=cv2.imread(path) # get the array of the original pic
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img) # original pic


    PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 70, 70, 3)
    print('transf img shape: ', PIC.shape)
    pred3 = model.predict(PIC)[0]
    print('Prediccion: ', pred3)
    return "Probs -> 1c:{0:.5f} 2c:{1:.5f} 5c:{2:.5f}".format(pred3[0],pred3[1],pred3[2])