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
	edges = cv2.Canny(img,100,200)
	image_copy = frame.copy()
	contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	cnt = contours[0]
	
	approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
	if len(approx) == 4:
		x, y, w, h = cv2.boundingRect(cnt)
		image_copy = cv2.drawContours(image_copy, [cnt], -1, (0,255,0), 3)
		return (x, y, x+w, y+h)

def localize_video(video):
	if video is None or len(video) == 0:
		return None

	h, w = video[0].shape[:2]
	rects = []
	for frame in video:
		screen = localize(frame)
		if screen is not None:
			rects.append(screen)
		#cv2.imshow('localized', frame)


	# find the most common rectangle
	# TODO: unhardcode values
	block_size = 4
	counterX = np.zeros((2, int(w/block_size) + 4))
	counterY = np.zeros((2, int(h/block_size) + 4))
	for rect in rects:
		counterX[0][int(rect[0]/block_size)] += 1
		counterX[1][int(rect[2]/block_size)] += 1

		counterY[0][int(rect[1]/block_size)] += 1
		counterY[1][int(rect[3]/block_size)] += 1

	x1 = np.argmax(counterX[0], axis=0) * block_size
	x2 = np.argmax(counterX[1], axis=0) * block_size
	y1 = np.argmax(counterY[0], axis=0) * block_size
	y2 = np.argmax(counterY[1], axis=0) * block_size

	return (x1, y1, x2, y2)

def is_valid_screen(screen, w, h):
	x1, y1, x2, y2 = screen
	# TODO: unhardcode values
	# only valid if the area is large enough and the aspect ratio is not too skewed
	return (x2-x1)*(y2-y1) > 0.2*w*h and min(x2-x1, y2-y1) > 0.3*max(x2-x1, y2-y1)

def find_video(video):
	if video is None or len(video) == 0:
		return None
	h, w = video[0].shape[:2]

	x_crop = [0, 0.1, 0.15, 0.25]
	y_crop = [0, 0.1, 0.15, 0.25]
	for x_ratio in x_crop:
		for y_ratio in y_crop:
			cropped_video = [video[i][int(x_ratio*h):int((1-x_ratio)*h), int(y_ratio*w):int((1-y_ratio)*w)] for i in range(0, len(video))]
			screen = localize_video(cropped_video)
			"""img = cropped_video[0].copy()
			cv2.rectangle(img, (screen[0], screen[1]), (screen[2], screen[3]), (0, 255, 0), 2)
			cv2.imshow('cropped', img)
			cv2.waitKey(0)"""
			if screen is not None and is_valid_screen(screen, w, h):
				return screen
	#return (x1, y1, x2, y2)

def read_video(query):
	cap = cv2.VideoCapture(query)
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	video = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			video.append(frame)
		else: 
			break

	cap.release()
	cv2.destroyAllWindows()
	return video

def show_video(query):
	screen = find_video(read_video(query))
	x1, y1, x2, y2 = None, None, None, None
	if screen is not None:
		x1, y1, x2, y2 = screen

	cap = cv2.VideoCapture(query)
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			if x1 is not None:
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.imshow('Frame', frame)

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
    