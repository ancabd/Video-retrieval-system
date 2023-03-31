import os
import sys
import numpy as np
import cv2
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default="Queries/")
	parser.add_argument('--sample_frequency', type=int, default=2)
	args = parser.parse_args()
	return args

# finds a screen given a frame
# returns coordinates (top left, top right, bottom left, bottom right)
def localize(frame):
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	#image_copy = frame.copy()
	#cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
	dst = cv2.cornerHarris(img,2,3,0.04)
	dst = cv2.dilate(dst,None)

	#frame[dst>0.01*dst.max()]=[0,0,255]

	#cv2.imshow('contours',image_copy)
	#cv2.waitKey(0)
	#return frame
	return harris_corner_detection(frame, 3, 0.01)

def harris_corner_detection(img, window_size, threshold):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        
    height, width = img.shape[:2]
    matrix_R = np.zeros((height,width))
    
    # Calculate the x and y image derivatives using Sobel operator
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    # dy, dx = np.gradient(gray)

    # Calculate product and second derivatives
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy

    offset = int( window_size / 2 )

    # Calculate second moment matrix for every point
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])

            #  Calculate second moment matrix for current point (x, y)
            M = np.array([[Sx2,Sxy],[Sxy,Sy2]])

            #  Approximate the product over the sum of the eigenvalues of the second moment matrix
            det, tr = np.linalg.det(M), np.matrix.trace(M)
            if tr == 0:
                matrix_R[y-offset, x-offset] = 0
            else:
                matrix_R[y-offset, x-offset] = det/tr
    
    # Apply a threshold
    cornerList = []
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value=matrix_R[y, x]
            if value>threshold:
                cornerList.append([x, y, value])
                cv2.circle(img,(x,y),1,(0,255,0))
    return img

def show_video(query):
	cap = cv2.VideoCapture(query)
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			cv2.imshow('Frame', localize(frame))

			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else: 
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	args = get_args()
	sample_frequency = args.sample_frequency
	path = args.input_path
	queries = os.listdir(path)

	for query in queries:
		show_video(path + query)
    