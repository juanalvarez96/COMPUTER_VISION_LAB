
def show_hist(hist):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(hist[i])
        cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv2_imshow(img)



def camshift_detection():
  selection = []
  track_window = []
  detected = []


  VideoCapture()
  eval_js('create()')
  while True: 
    byte = eval_js('capture()')
    im = byte2image(byte)
    vis = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # The first time. this "if" obtains histogram for the first time.
    # If no face was detected, new iteration and take new photo to detect the face.
    if selection == []:
      selection = detect(gray)
      #print(selection*)
      if selection == []:# we must detect a face or else we won't move forward
        continue
      else:
        #print(type(selection))
        x0, y0, x1, y1 = tuple(selection.reshape(1, -1)[0])
        # Select just the face region
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        # get distribution only one time of a face
        hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        hist = hist.reshape(-1)
        #Define track window to the face area
        track_window= (x0,y0, x1-x0, y1-y0)
        show_hist(hist) # We can omit this part

        vis_roi = vis[y0:y1, x0:x1]
        cv.bitwise_not(vis_roi, vis_roi)
        vis[mask==0]=0

    if track_window and track_window[2] > 0 and track_window[3] > 0:
      # We will come here for the next photo after detecting a face
      # Prob will contain the likely areas where the face is (according to histogram)
      prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
      prob &= mask      
      term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
      # Update track window to follow the new face area
      track_box, track_window = cv.CamShift(prob, track_window, term_crit)
      #print(track_box)
      xmin,ymin,dx,dy = track_window  
      detectedArea = [(xmin, ymin, xmin+dx, ymin+dy)]  
      draw_rects(prob, detectedArea, (255,0,0), 'Detected face')
      eval_js('showing("{}")'.format(image2byte(prob)))



    

camshift_detection()

