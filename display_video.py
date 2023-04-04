import os
from localization import find_video, read_video
import cv2
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default="Queries/")
	parser.add_argument('--sample_frequency', type=int, default=2)
	args = parser.parse_args()
	return args

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
    