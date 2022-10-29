import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

####################################
# Variable Declaration
# Width and height of the camera window
wCam, hCam = 640, 480
frameR = 100   # Frame Reduction
smoothening = 7
####################################

# 1. Find the hand landmarks
# 2. To get the tip of the inde
# 4. Only Index Finger : Movinx and middle finger
# # 3. Check which fingers are upg Mode
# 5. Convert coordinates to get correct position
# 6. Smoothen the values
# 7. Move Mouse
# 8. Both Index and Middle fingers are up, then it is clicking mode
# 9. Find distance between fingers
# 10. Click mouse if distance is short
# 11. Frame rate
# 12. Display

########################################
# Previous location
plocX, plocY = 0, 0

# Current location
clocX, clocY = 0, 0
########################################

cap = cv2.VideoCapture(0)

# Fixed width and height
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()               # Return the size of the screen
# print(wScr, hScr)

while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)

    # Getting the tip of the Index and Middle Finger
    if len(lmList)!= 0:
        x1, y1 = lmList[8][1:]            # Index Finger coordinates
        x2, y2 = lmList[12][1:]            # Middle Finger coordinates
        # print(x1, y1, x2, y2)

    # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # Box for mouse detection
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 0), 2)

        # if only the Index finger is up
        if fingers[1]==1 and fingers[2]==0:

            # Convert coordinates

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # Move Mouse
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)

            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            if length <35:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)

                # Clicking Mouse
                autopy.mouse.click()




    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
