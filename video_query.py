#!/usr/bin/env python
import argparse
import video_search
import numpy as np
import cv2
import glob
from scipy.io import wavfile
from video_tools import *
import feature_extraction as ft    
import sys
import os
from video_features import *
import pickle

features = ['colorhists', 'sift', 'all', 'mfccs', 'colorhistdiffs']
 
parser = argparse.ArgumentParser(description="Video Query tool")
#parser.add_argument("training_set", help="Path to training videos and wav files")
parser.add_argument("query", help="query video")
parser.add_argument("-s", help="Timestamp for start of query in seconds", default=0.0)
parser.add_argument("-e", help="Timestamp for end of query in seconds", default=0.0)
parser.add_argument("-f", help="Select features "+str(features)+" for the query ", default='colorhists')
parser.add_argument("-p", help="Select percantage of colorhist", default=50, type= int)
args = parser.parse_args()

if not args.f in features:
    print("Requested feature '"+args.f+"' is not a valid feature. Please use one of the following features:")
    print(features)
    

cap = cv2.VideoCapture(args.query)
frame_count = get_frame_count(args.query) + 1
frame_rate = get_frame_rate(args.query )
q_duration = float(args.e) - float(args.s)
q_total = get_duration(args.query)

if not float(args.s) < float(args.e) < q_total:
    print('Timestamp for end of query set to:', q_duration)
    args.e = q_total

# store the indicated features for every video frame
query_features = []
query_features1 = []
prev_frame = None
prev_colorhist = None
# starting frame number
frame_nbr = int(args.s)*frame_rate
# set the starting moment to play the video 
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < (int(args.e)*frame_rate)):
    ret, frame = cap.read()
    if frame is None:
        break

    if args.f == features[0]: 
        h = ft.colorhist(frame)
        if h is not None:
            query_features.append(h)
    elif args.f == features[1]:
        h = ft.sift(frame)
        if h is not None:
            query_features.append(h)
    elif args.f == features[2]:
        h_colorhist = ft.colorhist(frame)
        h_sift = ft.sift(frame)
        if h_colorhist is not None and h_sift is not None:
            query_features.append(h_colorhist)
            query_features1.append(h_sift)

    
    prev_frame = frame
    frame_nbr += 1

# Compare with database

video_types = ('*.mp4', '*.avi')
audio_types = ('*.wav', '*.WAV')

# grab all video file names
video_list = []
training_set = "experimentVideos"
for type_ in video_types:
    files = training_set + '/' +  type_
    video_list.extend(glob.glob(files))	

db_name = 'db/video_database.db'
print(db_name)
search = video_search.Searcher(db_name)

def sliding_window(x, w, compare_func):
    """ Slide window w over signal x. 
        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    frame = -1 # init frame 
    wl = len(w)
    minimum = sys.maxsize
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        if diff < minimum:
            minimum = diff
            frame   = i
    return frame, minimum

def sliding_window_max(x, y, len_w, compare_func):
    """ Slide window w over signal x. 
        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    maximum = -sys.maxsize -1
    for i in range(len(x) - len_w):
        for j in range(len(y) - len_w):
            diff = compare_func(x[i:(i+len_w)], y[j:(j+len_w)])
            if diff > maximum:
                maximum = diff
    return maximum
   
def euclidean_norm_mean(x,y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)   
    return np.linalg.norm(x-y)

def euclidean_norm(x,y):
    return np.linalg.norm(np.array(x)-np.array(y))

def prep_sift(x,w,func):
    fname = "db/base" + '_sift_vocabulary.pkl'
        # Load the vocabulary to project the features of our query image on
    with open(fname, 'rb') as f:
        sift_vocabulary = pickle.load(f, encoding="bytes")
    words = []
    for frame in w:
        words.append(np.array(sift_vocabulary.project(frame))) 
    #print(x)
    #x = [i / 20 for i in x]
    #print(len(x))
    #words = [i / 20 for i in words]
    
    return sliding_window(x,words, func)

def normalize_colorhist(score):
    return 1- (score/max_distance_colorhist)
def normalize_sift(score):
    return 1- (score/max_distance_sift)


top_3 =           { "[video1, sys.maxsize]" : -sys.maxsize-1,
                    "[video2, sys.maxsize]" : -sys.maxsize-1,
                    "[video3, sys.maxsize]" : -sys.maxsize-1
                    }

norm_colorhist = []
len_w = 0
def max_colorhist_dif_estimation():
    b_vid = np.zeros([480,640,3])
    w_vid = np.zeros([480,640,3])
    w_vid.fill(255)
    b_hist = ft.colorhist(b_vid)
    w_hist = ft.colorhist(w_vid)
    dif = euclidean_norm(b_hist, w_hist)/2
    max_distance_colorhist = dif * frame_count /frame_rate
    return max_distance_colorhist


max_distance_colorhist = max_colorhist_dif_estimation()
max_distance_sift = 7* frame_count/frame_rate

# Loop over all videos in the database and compare frame by frame
for video in video_list:
    print(video)
    if get_duration(video) < q_duration:
        print(get_duration(video), q_duration)
        print('Error: query is longer than database video')
        continue

    if args.f ==features[0] or args.f == features[1]:
        w = np.array(query_features)

    if args.f == features[2]:
        w = np.array(query_features)
        y = np.array(query_features1)
    
    if args.f == features[0]: 
        x = search.get_colorhists_for(video)
        frame, score = sliding_window(x,w, euclidean_norm_mean)
    
    elif args.f == features[1]:
        x = search.get_sift_for(video)
        frame, score = prep_sift(x,w, euclidean_norm)
    
    elif args.f == features[2]:
        len_w = len(w)
        x = search.get_colorhists_for(video)
        norm_colorhist.append(x)
        frame_colorhist, score_colorhist = sliding_window(x,w, euclidean_norm_mean)
        score_colorhist = normalize_colorhist(score_colorhist)   
        print(score_colorhist, video)
           
        x_sift = search.get_sift_for(video)
        frame_sift, score_sift= prep_sift(x_sift,y, euclidean_norm)
        
        score_sift = normalize_sift(score_sift)
        print(score_sift)

        score = (score_colorhist ** (args.p /100) * (score_sift ** ((100-args.p)/100)))
    
        min_hist = min(top_3, key=top_3.get)
        if score > top_3.get(min_hist):
            top_3.pop(min_hist)
            top_3[str(video) + "," + str(frame_colorhist/frame_rate)] = score
            

        
#        x = search.get_audiopowers_for(video)
#        frame, score = sliding_window(x,w, euclidean_norm)
#    elif args.f == features[3]:
#        x = search.get_mfccs_for(video)
        #frame, score = sliding_window(x,w, euclidean_norm_mean)
#        availableLength= min(x.shape[1],w.shape[1])
#        frame, score = sliding_window(x[:,:availableLength,:],w[:,:availableLength,:], euclidean_norm_mean)
    elif args.f == features[4]:
        x = search.get_chdiffs_for(video)
        frame, score = sliding_window(x,w, euclidean_norm)


sorted_scores= sorted(top_3.items(), key=lambda x:x[1] ,reverse=True)
dict_top_3 =dict(sorted_scores)
print(dict_top_3)