# Mehod to merely paint a white rectangle delimited by region on the image img
def draw_rects_filled(img, rects, color, Label):
    cv2.putText(img, Label, (rects[0][0], rects[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)


# The code in this method is practically the same as in lab2.
def camshift_face_removal():
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

    # The first time. this "if" obtains histogram for the first time
    if selection == []:
      selection = detect(gray)
      #print(selection*)
      if selection == []:# we must detect a face or else we won't move forward
        continue
      else:
        #print(type(selection))
        x0, y0, x1, y1 = tuple(selection.reshape(1, -1)[0])
        #Select the face
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        # get distribution only one time of a face
        hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        hist = hist.reshape(-1)
        # Update track window for the first time to where the face currently is
        track_window= (x0,y0, x1-x0, y1-y0)
        vis_roi = vis[y0:y1, x0:x1]
        cv.bitwise_not(vis_roi, vis_roi)
        vis[mask==0]=0

    if track_window and track_window[2] > 0 and track_window[3] > 0:
      prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
      prob &= mask
      # prob will contain the pixels where the face is more likely to be
      # we apply a filter using mask
      term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
      #update track window to where the face probably now is
      track_box, track_window = cv.CamShift(prob, track_window, term_crit)
      #print(track_box)
      xmin,ymin,dx,dy = track_window  
      detectedArea = [(xmin, ymin, xmin+dx, ymin+dy)]  
      # Paint the detected face
      draw_rects_filled(prob, detectedArea, (255,0,0), 'Removed face')
      eval_js('showing("{}")'.format(image2byte(prob)))

camshift_face_removal()

