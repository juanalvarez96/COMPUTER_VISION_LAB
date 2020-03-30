
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 
import cv2 as cv
from google.colab.patches import cv2_imshow # Needed to print out image

# OpenCv imports
from common import clock, draw_str
import video
from video import presets


import io
import base64



from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from PIL import Image

# Debugging
import pdb

# Histogram imports
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


#MAIN PROGRAM. Laboratory 1

'''
face detection using haar cascades

'''

# Detection algorithm import
# Load cascade file for the detection
cascade_fn="/content/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cv2.samples.findFile(cascade_fn))

def VideoCapture():
  js = Javascript('''
    async function create(){
      div = document.createElement('div');
      document.body.appendChild(div);

      video = document.createElement('video');
      video.setAttribute('playsinline', '');

      div.appendChild(video);

      stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}});
      video.srcObject = stream;

      await video.play();

      canvas =  document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      div_out = document.createElement('div');
      document.body.appendChild(div_out);
      img = document.createElement('img');
      div_out.appendChild(img);
    }

    async function capture(){
        return await new Promise(function(resolve, reject){
            pendingResolve = resolve;
            canvas.getContext('2d').drawImage(video, 0, 0);
            result = canvas.toDataURL('image/jpeg', 0.8);
            pendingResolve(result);
        })
    }

    function showing(imgb64){
        img.src = "data:image/jpg;base64," + imgb64;
    }

  ''')
  display(js)

# Needed to convert an image to array
def byte2image(byte):
  jpeg = base64.b64decode(byte.split(',')[1])
  im = Image.open(io.BytesIO(jpeg))
  return np.array(im)

#Inverse operation as above. Needed to show the image.
def image2byte(image):
  image = Image.fromarray(image)
  buffer = io.BytesIO()
  image.save(buffer, 'jpeg')
  buffer.seek(0)
  x = base64.b64encode(buffer.read()).decode('utf-8')
  return x
  # Function which draws a rectangle in a given photo. It can also select a color and add a label
def draw_rects(img, rects, color, Label):
    cv2.putText(img, Label, (rects[0][0], rects[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# This is the function responsible for detecting a face using the cascade classifier.
def detect(img, cascade = cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#Limit margin of the detected area. Set to 50
def setMargin(im, area, margin = 50):
    area[0][0] = max(area[0][0]-margin, 0)
    area[0][1] = max(area[0][1]-margin, 0)
    area[0][2] = min(area[0][2] + margin, im.shape[1])
    area[0][3] = min(area[0][3] + margin, im.shape[0])

#Transform the points from the previously cropped area
def transformation(rects, area):
    rects[0][0] += area[0][0]
    rects[0][1] += area[0][1]
    rects[0][2] += area[0][0]
    rects[0][3] += area[0][1]

# Initialize the video
VideoCapture()
eval_js('create()')


rects = []
area = []
while True:
  
  byte = eval_js('capture()')
  im = byte2image(byte)
  #rects = cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
  vis = im.copy()
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
  
  if (rects==[]):
    rects = detect(gray, cascade)
    # There is no face detected
    # We start over
    

  else:
    # Set a margin in case the image position has changed
    setMargin(im, area, 50)
    #Crop the image with the rectangle from before for detecting the next image in those boundaries.
    cropped_gray = gray[area[0][1]:area[0][3], area[0][0]:area[0][2]]
    rects = detect(cropped_gray, cascade)
    for (x1, y1, x2, y2) in area:
      draw_rects(vis, area, (255, 4, 0), "Rectangle")
      
    if rects != []:
      # Set points to the real position where they should be
      transformation(rects, area)

  area = rects

  
  
  vis = im.copy()
  
  for (x, y, w, h) in rects:
    draw_rects(vis, rects, (255, 4, 0), "Face!")
  
  
  eval_js('showing("{}")'.format(image2byte(vis)))