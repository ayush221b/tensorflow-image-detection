import tensorflow as tf
import sys
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from word2number import w2n
import requests
from pyfiglet import Figlet


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#Webcam code
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        out = cv2.imwrite('capture.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()
#till here

# image_path = sys.argv[1]

root = tk.Tk()
root.withdraw()

#image path changed
image_path = "/home/ayush17/Projects/mip-1/tensorflow-image-detection/capture.jpg"

if image_path:
    
    # Read the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        human_string=''

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            break

        f = Figlet(font='slant')
        print (f.renderText('< L U C I >'))
        
        #Printing detected indian currency
        inr=0
        inr = w2n.word_to_num(human_string)
        print("Detected INR:", inr)
        
        #printing USD value
        data= requests.get("http://apilayer.net/api/live?access_key=a59672cd83ac12ff7ce64f6a613c73f1&currencies=USD,%20INR&source=USD&format=1").json()
        for key, value in data.items():
            if key == "quotes":
                for key1, value1 in value.items():
                    usdinr=value1
        usd = inr/usdinr
        print('%s %.2f' % ("Value in USD (United States Dollar):", usd))

        #printing EUR value
        data= requests.get("http://apilayer.net/api/live?access_key=a59672cd83ac12ff7ce64f6a613c73f1&currencies=EUR,%20INR&source=USD&format=1").json()
        for key, value in data.items():
            if key == "quotes":
                for key1, value1 in value.items():
                    usdeur=value1
                    break
        eur = usd*usdeur
        print('%s %.2f' % ("Value in EUR (Euro):", eur))

        #printing CAD value
        data= requests.get("http://apilayer.net/api/live?access_key=a59672cd83ac12ff7ce64f6a613c73f1&currencies=CAD,%20INR&source=USD&format=1").json()
        for key, value in data.items():
            if key == "quotes":
                for key1, value1 in value.items():
                    usdcad=value1
                    break
        cad = usd*usdcad
        print('%s %.2f' % ("Value in CAD (Candian Dollars):", cad))

