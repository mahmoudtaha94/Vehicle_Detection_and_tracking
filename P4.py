from moviepy.editor import VideoFileClip

from step1_P4 import calibrate,undistort
from step2_P4 import getChannel,getGradient,applyBinary,combineAND,combineOR,stack2Channels
from step3_P4 import changePerspective
from step4_P4 import findLanes,drawLanes
from step5_P4 import curverad,centerDiff
from step6_P4 import writeOnFrame


def pipeline(img,mtx,dist):
    '''
    undistort images
    '''
    img=undistort(img,mtx,dist)
    
    '''
    prepare Images
    '''
    gray=getChannel(img,mode='gray',channel=None)
    gradx=getGradient(gray,mode='x',sobel_kernel=3)
    gradx=applyBinary(gradx,thresh=(20,255))
    
    grady=getGradient(gray,mode='y',sobel_kernel=3)
    grady=applyBinary(grady,thresh=(20,255))
    
    S=getChannel(img,mode='HLS',channel='S')
    sbinary=applyBinary(S,thresh=(100,255))
    
    V=getChannel(img,mode='HSV',channel='V')
    vbinary=applyBinary(V,thresh=(50,255))
    
    colorBinary=combineAND(vbinary,sbinary)
    
    image=combineAND(gradx,grady)
    image=combineOR(image,colorBinary)
    image[image==1]=255
    '''
    work on changing prespective 
    '''
    trapizoid={}
    trapizoid['first']=(590,445)
    trapizoid['second']=(690,445)
    trapizoid['third']=(1126,673)
    trapizoid['forth']=(153,673)
    warped=changePerspective(image,trapizoid,0.25,inverse=False)
    '''
    find the lanes
    '''
    My_ym=10/720
    My_xm=4/400
    leftx,rightx,midx=findLanes(warped,window_width=25,My_ym=My_ym, My_xm=My_xm,window_height=80)
    '''
    draw the lanes on the video frames
    '''
    result,res_yvals,yvals,left_fitx,right_fitx=drawLanes(img,warped,leftx,rightx,midx,trapizoid,window_width=25,window_height=80)
    '''
    calculate the curvature radius in meters based on a new line in the middle of the lane (brown)
    '''
    curve_rad=curverad(My_xm, My_ym,res_yvals,yvals,midx)
    '''
    calculate the difference from the center of the lane
    '''
    center_diff,side_pos=centerDiff(warped,left_fitx,right_fitx,My_xm)
    '''
    write the curvature radius and center difference on the video frames
    '''
    writeOnFrame(result,curve_rad,center_diff,side_pos)

    return result













