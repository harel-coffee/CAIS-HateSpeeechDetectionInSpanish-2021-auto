import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from .Dataset import Dataset

class DatasetAMI (Dataset):
    """
    DatasetAMI
    
    Unfortunately nowadays more and more episodes of harassments against women arose  and misogynistic 
    comments can be found in social media, where misogynists hide themselves behind the security of the 
    anonymity. Therefore, it is very important to identify misogyny in social media. Recent investigations 
    studied how the misogyny phenomenon takes place, for example as unjustified slurring or as stereotyping 
    of the role/body of a woman (i.e. the hashtag #getbacktokitchen), as described in the book by Poland [1]. 
    
    A preliminary research work was conducted by Hewitt et al. [2] as first attempt of manually classification 
    of misogynous tweets. So far it has not been carried out any attempt of automatically identifying misogynous 
    content in social media. The same shared tasks have been organized in occasion of IberEval-2018.
    
    Task A: Misogyny Identification
    Task B: Misogynistic Behaviour and Target Classification
    
    Traits:
        - Stereotype & Objectification: a widely held but fixed and oversimplified image or idea of a woman; 
          description of women’s physical appeal and/or comparisons to narrow standards.
        - Dominance: to assert the superiority of men over women to highlight gender inequality.
        - Derailing: to justify woman abuse, rejecting male responsibility; an attempt to disrupt the 
          conversation in order to redirect  women’s conversations on something more comfortable for men.
        - Sexual Harassment & Threats of Violence: to describe actions as sexual advances, 
          requests for sexual favors, harassment of a sexual nature; intent to physically assert power 
          over women through threats of violence.
        - Discredit: slurring over women with no other larger intention.
        - On the other, the target classification is again binary:

    Target:
        - Active (individual): the text includes offensive messages purposely sent to a specific target.
        - Passive (generic): it refers to messages posted to many potential receivers.
        
    Data:
    Two balanced datasets have been created for each collection, that is in Spanish and in English.  
    The data was manually labelled by two annotators according to three levels, namely Misogyny Identification, 
    Misogynistic Category Classification and Target Classification. Cases in disagreement were solved by a 
    third annotator.
    
    Results:
    Task A: https://amiibereval2018.files.wordpress.com/2018/05/spanish-subtaska.pdf
    Task B: https://amiibereval2018.files.wordpress.com/2018/05/spanish-subtaskb.pdf
    Full: https://amiibereval2018.files.wordpress.com/2018/05/spanish-detailed-results-category-target3.pdf
    
    @link https://amiibereval2018.wordpress.com/
    
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
        
        
        # Iterate over files
        for index, dataframe in enumerate (['train.csv', 'test.csv']):
        
            # @var corpus String
            file = self.get_working_dir ('corpus', dataframe)
        
        
            # @var df_split DataFrame
            df_split = pd.read_csv (file, delimiter = ';')
            
            
            # Determine training and testing splits
            df_split = df_split.assign (__split = 'train' if index == 0 else 'test')

            
            # Remove garbage columns
            if index == 0:
                df_split = df_split.drop (columns = ['id', 'twitter_id', 'training'])
            else:
                df_split = df_split.drop (columns = ['id'])
            
            
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)
        
        
        # @var val_index List Sample some training indexes 
        val_index = df[df['__split'] == 'train'].sample (frac = 0.25, replace = False).index.to_list ()
        
        
        # Change those to validation split
        df.loc[val_index, '__split'] = 'val'
        
        
        # NaNs for other tasks
        df.loc[df['target'] == '0', 'target'] = ''
        df.loc[df['misogyny_category'] == '0', 'misogyny_category'] = 'non_misogynous'
        
        
        # Reassign labels
        df = df.rename (columns = {
            'text': 'tweet', 
            'misogynous': 'label',
        })
        
        
        # Reassign labels as categories
        df.loc[df['label'] == 0, 'label'] = 'non_misogynous'
        df.loc[df['label'] == 1, 'label'] = 'misogynous'
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        
