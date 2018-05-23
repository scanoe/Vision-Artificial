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
from matplotlib.pyplot import title, imsave, hist
from skimage.color import rgb2gray
from skimage.color import rgb2yiq, rgb2hsv, rgb2xyz, rgb2lab, rgb2ycbcr
from skimage.filters import sobel_h, sobel_v,prewitt_h,prewitt_v,roberts


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

    



    
        
        
    
        

    
    
    
    
    
    