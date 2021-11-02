import config
import csv
import sys
import math
import string
import numpy as np
import pandas as pd
import os
import json

from tqdm import tqdm

from .Dataset import Dataset


class DatasetMisoCorpus (Dataset):
    """
        DatasetMisoCorpus
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
        
        
    def is_imabalanced (self, threshold = .15):
        """
        @inherit
        """
        return True
    
    def compile (self):
        
        # Load configuration of the dataset
        with open (self.get_working_dir ('dataset', 'MISOCORPUS.json')) as json_file:
            
            # @var df DataFrame
            df = pd.DataFrame (json.loads (json_file.read ())['data'])
            
            
            # Drop columns
            df = df.drop (labels = ['filtered'], axis = 1)
            df = df.rename (columns = {
                'class': 'label', 
                'twitter-id': 'twitter_id'
            })
            
            
            # Remove line-breaks
            df['tweet'] = df['tweet'].replace (r'\s+|\\n', ' ', regex = True)
            
            
            # Get random splits
            train, val, test = np.split (df.sample (frac = 1), [int (.6 * len (df)), int (.8 * len (df))])
            
            
            # Create NaN
            df = df.assign (__split = np.nan)
            
            
            # Assign 
            df['__split'][train.index] = 'train'
            df['__split'][val.index] = 'val'
            df['__split'][test.index] = 'test'
        
        # Return
        return df
        
