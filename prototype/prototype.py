import tkinter as tk

import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import sqlite3

def create_dp():
    database=sqlite3.connect('users.db')
    cursor=database.cursor()
    sql="""
        DROP TABLE IF EXISTS users;
        CREATE TABLE users (
                id integer unique primary key autoincrement,
                name text);
        """
    cursor.executescript(sql)
    database.commit()
    database.close()
    print("Dataset is created")

def register_face(username):
    database = sqlite3.connect('users.db')
    cursor = database.cursor()
    
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cap_video = cv2.VideoCapture(0)
    #u_name = input("Enter your name: ")
    
    cursor.execute('INSERT INTO users (name) VALUES (?)', (username,))
    given_id = cursor.lastrowid
    samp_num = 0
    
    while True:
      _,img = cap_video.read()
      grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      detectedfaces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
      for (x,y,w,h) in detectedfaces:
        samp_num = samp_num+1
        cv2.imwrite("dataset/User."+str(given_id)+"."+str(samp_num)+".jpg",grayscale[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.waitKey(100)
      cv2.imshow('img',img)
      cv2.waitKey(1);
      if samp_num > 25:
        break
    
    cap_video.release()
    database.commit()
    database.close()
    cv2.destroyAllWindows()
    
def train_model():
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    
    if not os.path.exists('./model'):
        os.makedirs('./model')
     
    def getImagesWithID(path):
      imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
      faces = []
      IDs = []
      for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
        
      return np.array(IDs), faces
  
    Ids, faces = getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save('model/trainingData.yml')
    cv2.destroyAllWindows()
    print("Training is done")
    return 1

def f_recognise():
    database = sqlite3.connect('users.db')
    cursor = database.cursor()
    fname = "model/trainingData.yml"
    if not os.path.isfile(fname):
      print("Please train the data first")
      exit(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap_video = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    while True:
      _,img = cap_video.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.2, 5)
      for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
        cursor.execute("select name from users where id = (?);", (ids,))
        result = cursor.fetchall()
        name = result[0][0]
        if conf < 50:
          cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
        else:
          cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      cv2.imshow('Face Recognizer',img)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
        break
    cap_video.release()
    cv2.destroyAllWindows()

def reg():
    canvas1 = tk.Canvas(root, width = 200, height = 200)
    canvas1.pack()
    label1 = tk.Label(root, text="Enter the name for the face")
    canvas1.create_window(100, 50, window=label1)
    entry1 = tk.Entry (root) 
    canvas1.create_window(100, 140, window=entry1)
    
    def dores():  
        x1 = entry1.get()
        print(x1)
        register_face(x1)
        canvas1.destroy()
    button1 = tk.Button(text='Enter', command=dores)
    canvas1.create_window(100, 180, window=button1)


if __name__ == '__main__':
    
    root=tk.Tk() 
    root.config(background='white') 
    root.title("Prototype v.1, author - vladarium")
    root.geometry("500x450") 
    
    fronttext = tk.LabelFrame(root, bg='white', bd=1, text="Prototype v.1, author - vladarium",font=14)                            
    fronttext.place(x=100, y = 20)  
    fronttext.pack()
    btn1 = tk.Button(root, text="Creat new database", command=create_dp, font=14)
    btn1.place(x=100, y=100)
    btn1.pack()
    
    btn2 = tk.Button(root, text="Start registring new face", command= reg, font=14)  
    btn2.place(x=100, y=150)
    btn2.pack()
    
    btn3 = tk.Button(root, text="Train your model based on received data", command=train_model, font=14)
    btn3.place(x=100, y=200)
    btn3.pack()
    
    btn4 = tk.Button(root, text="Try to recognise a face", command=f_recognise, font=14)
    btn4.place(x=100, y=250)
    btn4.pack()
    
    root.mainloop()   

    
                                   
