import cv2
'''
this file is step 6:
    1- draw on the video frames
'''
def writeOnFrame(result,curverad,center_diff,side_pos):
    cv2.putText(result,'Radius of Mid Curvature='+str(round(curverad,3))+'(m)',(300,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(300,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
