from scipy.cluster.vq import *
import numpy as np
import progressbar


class Vocabulary(object):
    """ The Vocabulary class takes a set of features and uses K-Means clustering to train a visual vocabulary. Then it can project new features on the vocabulary to obtain a histogram of visual words. """

    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, features, k=20, subsampling=10):
        """ Train a vocabulary from a dictionary of features 
                using k-means with k number of words. Subsampling 
                of training data can be used for speedup. """
                
        flat_list = [descs for vid in features for descs in vid]
        nbr_desc = len(flat_list)
        print(len(features[0][0][0]))
        print(len(features[0][0]))
        # stack all features for k-means
        # create empty array so we can stack new values against it
        descriptors = np.array([], dtype=np.float32).reshape(0,len(features[0][0][0]))
        for vid in features:
            for feat in vid:
                descriptors = np.vstack((descriptors, feat))

        #k-means
        self.voc, distortion = kmeans(descriptors[::subsampling,:],k,1)
        self.nbr_words = self.voc.shape[0]

        # go through all training images and project on vocabulary
        imwords = np.zeros((nbr_desc, self.nbr_words))        
        bar = progressbar.ProgressBar(maxval=nbr_desc, \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        count = 0
        for vid in features:    
            for desc in vid:
                bar.update(count)
                imwords[count] = self.project(desc)
                count += 1
        bar.finish()

        nbr_occurences = np.sum( (imwords > 0) * 1, axis=0)
        self.idf = np.log( (1.0*nbr_desc) / (1.0* nbr_occurences+1) )
        self.trainingdata = features



    def project(self, descriptors):
        """ project descriptors on the vocabulary
                to create a histogram of words"""
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1
        return imhist


