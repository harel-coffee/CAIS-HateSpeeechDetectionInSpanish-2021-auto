import csv
import sys
import string
import numpy as np
import pandas as pd
import os
import config

from .Dataset import Dataset

class DatasetHaterNet (Dataset):
    """
    DatasetHaterNet

    @link https://pubmed.ncbi.nlm.nih.gov/31717760/

    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
        
    
    def compile (self):
        
        # @var dfs List list of DataFrames
        dfs = []
        
        
        # @var corpus String
        file = self.get_working_dir ('corpus', 'corpus.txt')
        
        
        # @var df DataFrame
        df = pd.read_csv (file, sep = r';\|\|;', header = None, names = ['twitter_id', 'tweet', 'label'], engine="python")
        
        
        # Get splits
        df = self.assign_default_splits (df)
        
        
        # Fix ID
        df['twitter_id'] = df['twitter_id'].str.replace ('id=', '')
        
        
        # Reassign labels as categories
        df.loc[df['label'] == 0, 'label'] = 'non_hateful'
        df.loc[df['label'] == 1, 'label'] = 'hateful'
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        
        
    def get_columns_to_categorical (self):
        """ 
        {@inherit}
        """
        return ['__split', 'target', 'individual', 'aggresiveness']
        