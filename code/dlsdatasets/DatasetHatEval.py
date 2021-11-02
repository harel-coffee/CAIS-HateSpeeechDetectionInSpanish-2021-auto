import csv
import sys
import string
import numpy as np
import pandas as pd
import os
import config

from .Dataset import Dataset

class DatasetHatEval (Dataset):
    """
    DatasetHatEval
    
    Multilingual detection of hate speech against immigrants and women in Twitter (hatEval)
    
    Hate Speech is commonly defined as any communication that disparages a person or 
    a group on the basis of some characteristic such as race, color, ethnicity, 
    gender, sexual orientation, nationality, religion, or other characteristics. Given 
    the huge amount of user-generated contents on the Web, and in particular on 
    social media, the problem of detecting, and therefore possibly limit the Hate Speech diffusion, 
    is becoming fundamental, for instance for fighting against misogyny and xenophobia.
    
    The proposed task consists in Hate Speech detection in Twitter but featured by two specific 
    different targets, immigrants and women, in a multilingual perspective, for Spanish and English.
    
    TASK A
    Hate Speech Detection against Immigrants and Women: a two-class (or binary) classification 
    where systems have to predict whether a tweet in English or in Spanish with 
    a given target (women or immigrants) is hateful or not hateful.
    
    TASK B
    Aggressive behavior and Target Classification: where systems are asked first to classify hateful 
    tweets for English and Spanish (e.g., tweets where Hate Speech against women or immigrants has 
    been identified) as aggressive or not aggressive, and second to identify the target 
    harassed as individual or generic (i.e. single human or group).
    
    @link https://competitions.codalab.org/competitions/19935
    @link http://personales.upv.es/prosso/resources/BasileEtAl_SemEval19.pdf
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
        
    
    def compile (self):
        
        # @var corpus String
        file = self.get_working_dir ('corpus', 'full.csv')
        
        
        # @var df DataFrame
        df = pd.read_csv (file)
        
        
        # Filter by target
        if 'target' in self.options:
            df = df[(df.target == self.options['target'])]
        
        
        # Filter by language
        df = df[(df.language == self.options['hateval_lang_prefix'])]
        
        
        # Labels
        df["__split"] = np.nan
        df.loc[df['set'] == 'train', '__split'] = 'train'
        df.loc[df['set'] == 'dev', '__split'] = 'val'
        df.loc[df['set'] == 'test', '__split'] = 'test'
            
            
        # Reassign labels
        df = df.rename (columns = {
            "text": "tweet", 
            "HS": "label", 
            "AG": "aggresiveness",
            "TR": "individual"
        })
        
        
        # Remove useless columns
        df = df.drop (columns = ['id', 'language', 'set'])
        
        
        # Reassign labels as categories
        df.loc[df['label'] == 0, 'label'] = 'non_hatespeech'
        df.loc[df['label'] == 1, 'label'] = 'hatespeech'
        
        df.loc[df['aggresiveness'] == 0, 'aggresiveness'] = 'non_aggressive'
        df.loc[df['aggresiveness'] == 1, 'aggresiveness'] = 'aggresive'
        
        df.loc[df['individual'] == 0, 'individual'] = 'individual'
        df.loc[df['individual'] == 1, 'individual'] = 'collective'
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        
        
    def get_columns_to_categorical (self):
        """ 
        {@inherit}
        """
        return ['__split', 'target', 'individual', 'aggresiveness']
        