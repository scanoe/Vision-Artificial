# -*- coding: utf-8 -*-
"""
Created on Tue May 22 08:34:09 2018

@author: sebastian
"""

import scipy
import numpy as np
import skimage
import matplotlib
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from matplotlib.pyplot import imshow, show, subplot, figure
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel, prewitt
from matplotlib.pyplot import title, imsave, hist
from skimage.color import rgb2gray
from skimage.color import rgb2yiq, rgb2hsv, rgb2xyz, rgb2lab, rgb2ycbcr
from skimage.filters import sobel_h, sobel_v,prewitt_h,prewitt_v,roberts
from skimage.filters import threshold_otsu, threshold_adaptive
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, subplot, figure


cantBuenasA = 17 # numero de imagenes buenas del lado A para entrenar
cantBuenasB = 17 #cantidad de imagenes buenas para entrenar del lado B

cantMalasA =17  # numero de imagenes malas del lado A para entrenar 
CantMalasB=15  #cantidad de imagenes malas para entrenar del lado B

#Esta funcion Carga las imagenes del lado A para buenas y malas el lado A es
# el que tiene la linea en la mitad y es algo plano

def cargarimgA() :
    
    aux1=[]
    for i in range(cantBuenasA):
        aux1.append(imread("Bueno_A"+str(i+1)+".jpg"))
        
    Buenas=np.array(aux1)
    aux1.clear()
    
    for i in range(cantMalasA):
        aux1.append(imread("Malo_A"+str(i+1)+".jpg"))
        
    Malas=np.array(aux1)
    print("ok")
    return Malas , Buenas

def cargarimgB() :
    
    aux1=[]
    for i in range(cantBuenasB):
        aux1.append(imread("Bueno_B"+str(i+1)+".jpg"))
        
    Buenas=np.array(aux1)
    aux1.clear()
    
    for i in range(CantMalasB):
        aux1.append(imread("Malo_B"+str(i+1)+".jpg"))
        
    Malas=np.array(aux1)
    print("ok")
    return Malas , Buenas


def MostrarA (Malas,Buenas):
    cont=0
    for i in range(cantBuenasA):
        figure(cont)
        imshow(Buenas[i])
        title("Buena "+str(i))
        cont=cont+1
        
    for i in range(cantMalasA):
        figure(cont)
        imshow(Malas[i])
        title("Mala "+str(i))
        cont =cont+1

def MostrarB (Malas,Buenas):
    cont=0
    for i in range(cantBuenasB):
        figure(cont)
        imshow(Buenas[i])
        title("Buena "+str(i))
        cont=cont+1
        
    for i in range(CantMalasB):
        figure(cont)
        imshow(Malas[i])
        title("Mala "+str(i))
        cont =cont+1       
        
def crop(img):
    
    nimg=img[1480:2700,2000:3200]
    
    return nimg
    
def CropA(Malas,Buenas):
    
    aux1=[]
    
    for i in range(cantBuenasA):
        aux1.append(crop(Buenas[i]))
        
    cropedBuenasA=np.array(aux1)
    aux2=[]
    for i in range(cantMalasA):
        aux2.append(crop(Malas[i]))
    cropedMalasA=np.array(aux2)
    print("cropA ok")
    return cropedBuenasA,cropedMalasA

def CropB(Malas,Buenas):
    
    aux1=[]
    
    for i in range(cantBuenasB):
        aux1.append(crop(Buenas[i]))
        
    cropedBuenasB=np.array(aux1)
    aux2=[]
    for i in range(CantMalasB):
        aux2.append(crop(Malas[i]))
    cropedMalasB=np.array(aux2)
    print("cropA ok")
    return cropedBuenasB,cropedMalasB

## se sa para hacerel analisis de los espacios de color rgb, yiq hsv xyz lab yy cbcr
def pruebacolor(img):
    print("rgb")
    figure(0)
    imshow(img[:,:,0])
    title("rgb r")
    figure(1)
    imshow(img[:,:,1])
    title("rgb g")
    figure(2)
    imshow(img[:,:,2])
    title("rgb b")
    yiq=rgb2yiq(img)
    print("yiq")
    figure(3)
    imshow(yiq[:,:,0])
    title("yiq y")
    figure(4)
    imshow(yiq[:,:,1])
    title("yiq i")
    figure(5)
    imshow(yiq[:,:,2])
    title("yiq q")
    hsv=rgb2hsv(img)
    print("hsv")
    figure(6)
    imshow(hsv[:,:,0])
    title("hsv h")    
    figure(7)
    imshow(hsv[:,:,1])
    title("hsv s")
    figure(8)
    imshow(hsv[:,:,2])
    title("hsv v")
    xyz=rgb2xyz(img)
    print("xyz")
    figure(9)
    imshow(xyz[:,:,0])
    title("xyz x")
    figure(10)
    imshow(xyz[:,:,1])
    title("xyz y")
    figure(11)
    imshow(xyz[:,:,2])
    title("xyz z")
    lab=rgb2lab(img)
    print("lab")
    figure(12)
    imshow(lab[:,:,0])
    title("lab l")
    figure(13)
    imshow(lab[:,:,1])
    title("lab a")
    figure(14)
    imshow(lab[:,:,2])
    title("lab b")
    ycbcr =rgb2ycbcr(img)
    print("ycbcr")
    figure(15)
    imshow(ycbcr[:,:,0])
    title("ycbcr y")
    figure(16)
    imshow(ycbcr[:,:,1])
    title("ycbcr cb")
    figure(17)
    imshow(ycbcr[:,:,2])
    title("ycbcr cr")


def obtenerlabbA(Malas,Buenas):
    aux1=[]
    for i in range(cantBuenasA):
        img=rgb2lab(Buenas[i])[:,:,2]
        aux1.append(np.median(img))
    Buenoslabb=np.array(aux1)
    aux2=[]
    for i in range(cantMalasA):
        img=rgb2lab(Malas[i])[:,:,2]
        aux2.append(np.median(img))
    Maloslabb=np.array(aux2)
    
    return Maloslabb,Buenoslabb

def obtenerlabbB(Malas,Buenas):
    aux1=[]
    for i in range(cantBuenasB):
        img=rgb2lab(Buenas[i])[:,:,2]
        aux1.append(np.median(img))
    Buenoslabb=np.array(aux1)
    aux2=[]
    for i in range(CantMalasB):
        img=rgb2lab(Malas[i])[:,:,2]
        aux2.append(np.median(img))
    Maloslabb=np.array(aux2)
    
    return Maloslabb,Buenoslabb


def obtenerlab(img):
    img1=rgb2lab(img)[:,:,2]
    media=np.median(img1)
    return media

def border_thing2(img):
    # Array ya está recortado
    img = np.array(img, dtype=np.float64);
    #img = img[1480:2700,2000:3200];
    img_g = rgb2gray(img);
    
    # aplicación de filtros
    for c in range(0, 10):
        img_g = gaussian(img_g, 3);
    
    img_prw = prewitt(img_g, mask = None);
    #figure();
    #imshow(img_prw, cmap = "gray");
    #title("prewitt");
    '''
    fil, col, can = img.shape;
    for c in range(0, fil):
        for j in range(0, col):
            if(img_prw[c][j] < 1):
                img_prw[c][j] = 0;
    
    img_prw = img_prw*255;
    figure();
    imshow(img_prw, cmap = "gray");
    title("prewitt después");'''
    '''
    fil, col, can = img.shape;
    for c in range(0, fil):
        for j in range(0, col):
            if(img_prw[c][j] > 1):
                img_prw[c][j] = 1;
            else:
                img_prw[c][j] = 0;'''
                
    us_img = 255*(img_prw < 0.67).astype("uint8");
    #figure();
    #imshow(us_img, cmap = "gray");
    #title("umbralización");
    
    # conteo
    count = 0;
    fil, col, can = img.shape;
    for c in range(0, fil):
        for j in range(0, col):
            if(us_img[c][j] == 0):
                count += 1;
    
    return count;

def border_thing(Array, n):
    # Array ya está recortado
    res = [];
    for joder in range(n):
        img = Array[joder];
        img_g = rgb2gray(img);
        
        # aplicación de filtros
        for k in range(0, 10):
            img_g = gaussian(img_g, 3);
        
        img_prw = prewitt(img_g, mask = None);
        #figure();
        #imshow(img_prw, cmap = "gray");
        #title("prewitt");
        #figure();
        #hist(img_prw.ravel(), 256, [0,256]);
        
        us_img = (255*img_prw < 0.67).astype("uint8");
        figure();
        #imshow(us_img, cmap = "gray");
        #title("umbralización");
        
        # conteo
        count = 0;
        fil, col, can = img.shape;
        for c in range(0, fil):
            for j in range(0, col):
                if(us_img[c][j] == 0):
                    count += 1;
                    
        res.append(count);
        
    return res;

def entrenar(cara):
    
    if (cara=="a"or cara=="A"):
         malas,buenas=cargarimgA()
         cropbuenas,cropmalas=CropA(malas,buenas)
         bmala,bbuena=obtenerlabbA(cropmalas,cropbuenas)
         brodmalas=border_thing(cropmalas,cantMalasA)
         brodbuenas=border_thing(cropbuenas,cantBuenasA)
         traindata=[]
         responses=[]
         for i in range(cantBuenasA):
             traindata.append([bbuena[i],brodbuenas[i]])
             responses.append(0)
             
         for i in range(cantMalasA):
            traindata.append([bmala[i],brodmalas[i]])
            responses.append(1)
            

         
    elif (cara=="b" or cara=="B"):
         malas,buenas=cargarimgB()
         cropbuenas,cropmalas=CropB(malas,buenas)
         bmala,bbuena=obtenerlabbB(cropmalas,cropbuenas)
         brodmalas=border_thing(cropmalas,CantMalasB)
         brodbuenas=border_thing(cropbuenas,cantBuenasB)
         traindata=[]
         responses=[]
         for i in range(cantBuenasB):
             traindata.append([bbuena[i],brodbuenas[i]])
             responses.append(0)
             
         for i in range(CantMalasB):
            traindata.append([bmala[i],brodmalas[i]])
            responses.append(1)
    Train=np.array(traindata).astype(np.float32)
    respon=np.array(responses).astype(np.float32)
    print(Train)
    print(respon)
    knn = cv.ml.KNearest_create()
    knn.train(Train, cv.ml.ROW_SAMPLE, respon)
        
    return knn

def clasificar(Knn,imagen):
    img=imread(imagen)
    cropimg=crop(img)
    imgb=obtenerlab(cropimg)
    brod=border_thing2(cropimg)
    
    newcomer=np.array([[imgb,brod]]).astype(np.float32)
    print(newcomer)
    ret, results2, neighbours2 ,dist = Knn.findNearest(newcomer, 2)
    
    return results2



print("ingrese una cara A=cara plana o B:cara curva")
cara=input()

print("ingrese el archivo incluyendo el .jpg este debe estar en la carpeta")
imagen=input()

print("porfavor espere el proceso puede tardar unos minutos")
knn=entrenar(cara)
respuesta=clasificar(knn,imagen)


if (respuesta==0):
    print("la imagen es buena")
elif(respuesta==1):
    
    print("la imagen es mala")
    




    
    
    
    
    

     

            
            
             
             
         
         
         
         
         
        
        
        
    
    

