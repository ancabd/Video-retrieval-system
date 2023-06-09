#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import pickle
import glob
import feature_extraction as ft    
from scipy.io.wavfile import read
from video_tools import *
from video_features import *
import db_index_video
import Vocabulary




def create_database():
    # DATABASE Creating and insertion
    # If the database already exsists, we can remove it and recreate it, or we can just insert new data.
    db_name = 'db/video_database.db'

    if not os.path.exists('db'):
        os.makedirs('db')

    # check if database already exists
    new = False
    if os.path.isfile(db_name):
        action = input('Database already exists. Do you want to (r)emove, (a)ppend or (q)uit? ')
        print('action =', action)
    else:
        action = 'c'

    if action == 'r':
        print('removing database', db_name , '...')
        os.remove(db_name)
        new = True

    elif action == 'a':
        print('appending to database ... ')

    elif action == 'c':
        print('creating database', db_name, '...')
        new = True

    else:
        print('Quit database tool')
        sys.exit(0)

    # Create indexer which can create the database tables and provides an API to insert data into the tables.
    indx = db_index_video.Indexer(db_name) 
    if new == True:
        indx.create_tables()
    
    return indx
#
# Processing of videos
def process_videos(video_list, indx):
    base = "base"
    total = len(video_list)
    sifts = []
    sift_all = []
    colorhists = []
    for video in video_list:
        cap = cv2.VideoCapture(video)
        #frame_rate = get_frame_rate(video) 
        #total_frames = get_frame_count(video)
        #total_audio_frames = get_frame_count_audio(video)
        
        # get corresponding audio file
        #filename, fileExtension = os.path.splitext(video)

        colorhist = []
        sift_descriptors = []
        frame_nbr = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            colorhist_frame = ft.colorhist(frame)
            colorhist.append(colorhist_frame)
            desc = ft.sift(frame)
            if desc is not None:
                sift_descriptors.append(desc)
            #prev_frame = frame
            frame_nbr += 1
        sifts.append(sift_descriptors)
        colorhists.append(colorhist)
        
        # prepare descriptor for database
        # mfccs = descr['mfcc'] # Nx13 np array (or however many mfcc coefficients there are)
        # audio = descr['audio'] # Nx1 np array
        # colhist = descr['colhist'] # Nx3x256 np array
        # tempdif = descr['tempdiff'] # Nx1 np array
        print("processed the video", video)
    sift_all = np.array(sifts).flatten()
    sift_vocabulary = Vocabulary.Vocabulary(base)
    sift_vocabulary.train(sifts, 20)
    fname = "db/" + base + '_sift_vocabulary.pkl'
    with open(fname, 'wb') as f: 
        pickle.dump(sift_vocabulary,f)

    for i, video in enumerate(video_list):
        print(i)
        descr = {}
        descr['colhist'] = np.array(colorhists[i])
        descr['sift'] = np.array(sifts[i])
        indx.add_to_index(video,descr, sift_vocabulary)
    indx.db_commit()


    
parser = argparse.ArgumentParser(description="Video Processing tool extracts features for each frame of video and for its corresponding audio track")
parser.add_argument("training_set", help="Path to training videos and wav files")

args = parser.parse_args()


video_types = ('*.mp4', '*.avi')
audio_types = ('*.wav', '*.WAV')


# grab all video file names
video_list = []
for type_ in video_types:
    files = args.training_set + '/' +  type_
    print(files)
    video_list.extend(glob.glob(files))	
print(video_list)
# create database
indx = create_database()
process_videos(video_list,indx)
