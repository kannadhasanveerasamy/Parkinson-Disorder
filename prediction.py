import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image as keras_image

import numpy as np
import easygui
from keras.models import load_model
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
my_w = tk.Tk()
my_w.geometry('1244x829+0+10')  
my_w.title('Lungs  Prediction')
my_font1=('times', 18, 'bold')


image = Image.open("image4.png")
new_width = 1800
new_height = 1600
resized_image = image.resize((new_width, new_height))
resized_image.save("resize.png")
bg = Image.open('resize.png')
bg = ImageTk.PhotoImage(bg)
bgLabel = Label(my_w, image=bg)
bgLabel.place(x=0, y=0)

                 

l1 = tk.Label(my_w,text='Upload Files & get results',width=30,font=my_font1,bg='#000080',
                   fg='red',)  
l1.place(x=550, y=190, width=300)
b1 = tk.Button(my_w, text='Upload Images', 
   width=20,command = lambda:result(), activebackground='#000080', bg='green')
b1.place(x=590,y=500, width=230, height=40)

print(tf.__version__)

def close():
   my_w.destroy()


titleLabel = Label(my_w, text=' HISTOPOTHOLOGY PREDICTION', font=('italic', 22, 'bold '), bg='black',
                   fg='white', )
titleLabel.place(x=0, y=40, width=1350, height=50)

endbtn=Button(my_w,text="Exit",font='italic 14 bold',bg='black',fg='white',command=close)
endbtn.place(x=670,y=600,width=50)


classifierLoad = tf.keras.models.load_model('model2.h5')


def result():
    
   filename =upload_file()
   test_image2 = keras_image.load_img(filename, target_size = (200,200))
   test_image2 = keras_image.img_to_array(test_image2)
   test_image2 = np.expand_dims(test_image2, axis = 0)   
   # cnn prediction on the test image
   result = classifierLoad.predict(test_image2)
   print(result)
   if result[0][1] == 1:
       prediction2="colon2"
   elif result[0][0] == 1:
       prediction2="colon1"
   elif result[0][2] == 1:
       prediction2="lungs1"
   elif result[0][3] == 1:
       prediction2="lungs2"
   elif result[0][4] == 1:
       prediction2="lungs3"
   
      
   print(prediction2)
   prediction=prediction2
   l2 = tk.Label(my_w,text="Result :  "+prediction,width=50,font=my_font1,bg='pink',
                   fg='black',)  
   l2.place(x=560, y=550, width=400)
   return filename 
       
def upload_file():   
    filename =  easygui.fileopenbox()
    img=Image.open(filename) # read the image file
    img=img.resize((200,140)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 =tk.Label(my_w)
    e1.place(x=590, y=240, width=240, height=250)
    e1.image = img
    e1['image']=img
    return filename

my_w.mainloop()






