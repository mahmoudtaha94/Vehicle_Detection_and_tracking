import pickle
import glob

'''
this file extract the car/noncar images to be used to extract the project parameters
'''

# Read in cars and notcars
images = glob.glob('./non-vehicles/*.png')
cars = []
notcars = []
for image in images:
	notcars.append(image)

images = glob.glob('./vehicles/GTI_Far/*.png')
for image in images:
	cars.append(image)

images = glob.glob('./vehicles/KITTI_extracted/*.png')
for image in images:
	cars.append(image)
images = glob.glob('./vehicles/GTI_Left/*.png')
for image in images:
	cars.append(image)
images = glob.glob('./vehicles/GTI_MiddleClose/*.png')
for image in images:
	cars.append(image)
images = glob.glob('./vehicles/GTI_Right/*.png')
for image in images:
	cars.append(image)

imgs={}
imgs['notcars']=notcars
imgs['cars']=cars
with open("images.p", "wb") as output_file:
	pickle.dump(imgs, output_file)