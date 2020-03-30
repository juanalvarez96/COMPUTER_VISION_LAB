
#Method to extend (margin) region of a box (area) inside an image (im)
def amplifyArea(im, area, margin):
    new_area = [0,0,0,0]
    new_area[0] = max(area[0]-margin, 0)
    new_area[1] = max(area[1]-2*margin, 0)
    new_area[2] = min(area[2] + 2*margin, im.shape[1])
    new_area[3] = min(area[3] + 2*margin, im.shape[0])
    return new_area

#Mehtod to pain the black box
def draw_rects_filled(img, rects, color, Label):
    cv2.putText(img, Label, (rects[0][0], rects[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)

#Method to store the image in the current working directory
def storeHand(hand):
  cv2.imwrite('image.png', hand)


def camshift_detection():
  selection = []
  track_window = []
  detected = []
  #detectedArea = []
  detectedface = []
  amplified_margin = 50
  hsv_roi_selection = []
  vis_roi_selection = []
  black_box_coordinates = []
  track_window_hand = []
  face_size = []
  first_time = True

  VideoCapture()
  eval_js('create()')
  while True: 
    first_time = False
    #pdb.set_trace()
    byte = eval_js('capture()')
    im = byte2image(byte)
    vis = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Mask is configured to do a more "aggresive" filter to the hsv. 
    # Both for the face and the complete image
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    # Detect a face
    if selection == [] :
      # Detect face
      selection = detect(gray)
      if selection == [] : 
        # Don't do anyhting until face detected
        continue
      else:
        # Face detected
        #pdb.set_trace()
        first_time = True
        selection = selection[0]
        # Coordinates of face
        x0,y0,x1,y1 = tuple(selection)
        face_size = (x0, y0, x1-x0, y1-y0)
        # Get face distribution (histogram). Crop image to get just the face
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        hist = hist.reshape(-1)
        # Set black box area in face
        drawing_area = (x0, y0, x1, y1)
        #pdb.set_trace()
        # Track window face is where we will search for the NEXT face position          
        track_window_face = (x0,y0, x1-x0, y1-y0)
        prob_face = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
        prob_face &= mask
        term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
        track_box_face, track_window_face = cv.CamShift(prob_face, track_window_face, term_crit)
        xmin,ymin,dx,dy = track_window_face
        black_box_coordinates = (xmin, ymin, xmin+dx, ymin+dy)
        rects = amplifyArea(hsv, black_box_coordinates, 30)
        draw_rects_filled(hsv, [rects], (0, 0, 0), 'Black box')
        #pdb.set_trace()

        
    

    if first_time == False:
      #pdb.set_trace()
      
      # Update black box coordinate for next photo
      new_selection = amplifyArea(hsv, black_box_coordinates, 50)
      x0,y0,x1,y1 = tuple(new_selection)
      track_window_face = (x0,y0, x1-x0, y1-y0)
      #pdb.set_trace()
      hsv_face = hsv[y0:y1, x0:x1]
      mask_face = cv2.inRange(hsv_face, np.array((0., 60., 20.)), np.array((180., 180., 150.)))      
      #db.set_trace()
      mask_face = mask[y0:y1, x0:x1]
      prob_face = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
      prob_face &= mask
      #prob_face = cv.calcBackProject([hsv_face], [0], hist, [0, 180], 1)
      #prob_face &= mask_face
      term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
      track_box_face, track_window_face = cv.CamShift(prob_face, track_window_face, term_crit)
      xmin,ymin,dx,dy = track_window_face
      #pdb.set_trace()
      # Check the box is not too big
      if (track_window_face[2]>face_size[2]+20 or track_window_face[3]>face_size[3]+20):
        selection = []
        #print("start over")
        continue
        # We need to recalculate the face
      black_box_coordinates = (xmin, ymin, xmin+dx, ymin+dy)
      
      draw_rects_filled(hsv, [black_box_coordinates], (0, 0, 0), 'Black box')
      

    # Now we analyze the image (with the face covered) and look for the hand
    
    prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
    prob &= mask
    #Initially we don't know where will the user place the hand
    #Therefore the track _Window_hand will be the whole image (with the face covered)
    if track_window_hand == []:
      track_window_hand = (0,0,vis.shape[0], vis.shape[1])
    else:
      track_window_hand = amplifyArea(vis, track_window_hand, 50)
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
    track_box, track_window_hand = cv.CamShift(prob, track_window_hand, term_crit)
    xmin,ymin,dx,dy = track_window_hand
    detectedArea = (xmin, ymin, xmin+dx, ymin+dy)
    
    #Now we move on to painting the hand using the face's histogram
    #pdb.set_trace()
    draw_rects(prob, [detectedArea], (255,0,0), 'Detected hand')

    #Uncomment following line if you want to store hand
    #storeHand(im[detectedArea[1]:detectedArea[3], detectedArea[0]:detectedArea[2]])
    eval_js('showing("{}")'.format(image2byte(prob)))

    






camshift_detection()

