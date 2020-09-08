import urllib.request
from collections import deque
from scipy.spatial import distance as dist
# deque  a list like data structure to keep track of x,y of the current object

from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,help="max buffer size")
# more points being tracked
# buffer
args = vars(ap.parse_args())




#greenLower = (29, 86, 6)
# Yellow Color Object
Lower = (23,26,183)
Upper = (255,255,255)
#greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
# allow the camera or video file to warm up
time.sleep(2.0)

# uses IP-Webcam Camera App as Subsitution for Webcam
URL = "http://10.173.68.180:8080/shot.jpg"
# have to renew the url for the IP-Webcam

while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    if frame is None:
        break
        # resize the frame, blur it, and convert it to the HSV
        # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
       # ((x, y), radius) = cv2.minEnclosingCircle(c)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype= "int")
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #centroids
        # only proceed if there is a certain size
        if cv2.contourArea(c) < 100 :
            continue
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
           # cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # update the points queue
    pts.appendleft(center)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        if counter >= 10 and i == 1 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ("", "")
            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                dirX = "Left" if np.sign(dX) == 1 else "Right"
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                dirY = "Up" if np.sign(dY) == 1 else "Down"
            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    cv2.putText(frame, direction, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX,dY),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "{:.1f}px".format(dA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(frame, "{:.1f}px".format(dB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
# if we are not using a video file, stop the camera video stream
# otherwise, release the camera
# close all windows
cv2.destroyAllWindows()

