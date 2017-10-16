import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from Functions import *
'''
This file is preparing the project parameters:
1- features parameters
2- train the classifier
3- define the standard scaler

'''
img_pickle = pickle.load( open("images.p", "rb" ) )
notcars=[]
cars=[]
notcars=img_pickle['notcars']
cars=img_pickle['cars']
'''
lets train a classifier
'''
#define feaature parameters
ColorSpace='YCrCb'
orient=9
PixPerCell=8
CellPerBlock=2
HogChannel='ALL'
SpatialSize=(32,32)
HistBins=32
SpatialFeatures=True
HistFeatures=True
HogFeatures=True

test_cars=cars
test_notcars=notcars


car_features=extractFeatures(test_cars, ColorSpace=ColorSpace,SpatialSize=SpatialSize,HistBins=HistBins,
					orient=orient,PixPerCell=PixPerCell,CellPerBlock=CellPerBlock,HogChannel=HogChannel,SpatialFeatures=SpatialFeatures,
					HistFeatures=HistFeatures,HogFeatures=HogFeatures)
notcar_features=extractFeatures(test_notcars, ColorSpace=ColorSpace,SpatialSize=SpatialSize,HistBins=HistBins,
					orient=orient,PixPerCell=PixPerCell,CellPerBlock=CellPerBlock,HogChannel=HogChannel,SpatialFeatures=SpatialFeatures,
					HistFeatures=HistFeatures,HogFeatures=HogFeatures)


X=np.vstack((car_features,notcar_features)).astype(np.float64)
#fit a per-column scaler
X_scaler=StandardScaler().fit(X)


#apply scaler to x
scaled_X=X_scaler.transform(X)

#define the labels vector
y=np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))

#split the data into randomized training and testing data
rand_state=np.random.randint(0,100)
X_train,X_test,y_train,y_test=train_test_split(scaled_X,y,test_size=0.1,random_state=rand_state)


#use a linear SVC
svc=LinearSVC()
svc.fit(X_train,y_train)
print('the accuracy=',round(svc.score(X_test,y_test),4))
#save the trained classifier, the standard scaler, and feature parameter 
#to a pickle so that i don't have to make them every time i try a combination
ProjectParameters={}
ProjectParameters['SVCmodel']=svc
ProjectParameters['X_scaler']=X_scaler
ProjectParameters['ColorSpace']=ColorSpace
ProjectParameters['orient']= orient
ProjectParameters['PixPerCell']=PixPerCell
ProjectParameters['CellPerBlock']=CellPerBlock
ProjectParameters['HogChannel']=HogChannel
ProjectParameters['SpatialSize']=SpatialSize
ProjectParameters['HistBins']=HistBins
ProjectParameters['SpatialFeatures']=SpatialFeatures
ProjectParameters['HistFeatures']=HistFeatures
ProjectParameters['HogFeatures']=HogFeatures
with open("ProjectParameters.p", "wb") as output_file:
	pickle.dump(ProjectParameters, output_file)

