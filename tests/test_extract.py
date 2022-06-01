#!/usr/bin/env python
# coding: utf-8

# # Using Sherlock out-of-the-box
# This notebook shows how to predict a semantic type for a given table column.
# The steps are basically:
# - Download files for word embedding and paragraph vector feature extraction (downloads only once) and initialize feature extraction models.
# - Extract features from table columns.
# - Initialize Sherlock.
# - Make a prediction for the feature representation of the column.

# In[1]:


DIR_DATASET = '/ivi/inde/mmargaret/data_search_e_data_csv/'
DIR_LOG = '/ivi/inde/mmargaret/sherlock-project/log/'
DIR_OUTPUT = '/ivi/inde/mmargaret/sherlock-project/output/'

# DIR_DATASET = '/Users/mmargaret/Documents/[UVA] Thesis/sherlock-project/data/data_search_e_data_csv/'
# DIR_LOG = '/Users/mmargaret/Documents/[UVA] Thesis/sherlock-project/log/'
# DIR_OUTPUT = '/Users/mmargaret/Documents/[UVA] Thesis/sherlock-project/output/'


# In[2]:


import logging
from datetime import datetime
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='{}{}'.format(DIR_LOG,datetime.now().strftime('%Y%m%d_%H%M_sherlock.log')), mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

logging.info('START')


import numpy as np
import pandas as pd
import pyarrow as pa
import os

from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings


# ## Initialize feature extraction models

# In[4]:


prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()


# ## Extract features

_ = os.listdir(DIR_DATASET)


file_list = [id for id in _ if '.csv' in id]
logging.info('Number of Files: {}'.format(len(file_list)))
print('Number of Files: {}'.format(len(file_list)))


error_list=[]

def extractIDSemanticsWithColumnNames(filename):
    
    IDSemanticsColumns = {'data_filename':filename, 'colSemantics': [], 'colNames':[]}
    try:
        # read files
        with open(DIR_DATASET + filename, errors='ignore') as f:
            a_doc = pd.read_csv(f)
        a_doc = a_doc.astype(str) #only non-numeric object to str (sherlock required) = .select_dtypes(include=[object])
        data = pd.Series(a_doc.transpose().values.tolist(), name="values") #format it to list of values by columns

        # sherlock extract features
        extract_features("../temporary.csv",data)
        feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)

        # sherlock init and predict with pre-trained model
        model = SherlockModel();
        model.initialize_model_from_json(with_weights=True, model_id="sherlock");
        predicted_labels = model.predict(feature_vectors, "sherlock")

        # return dictionary with id: id of the doc, list of the columns' semantics, list of the columns' names
        IDSemanticsColumns = {'data_filename':filename, 'colSemantics': list(predicted_labels), 'colNames':list(a_doc.columns)}
    
    except Exception as e:
        logging.error(e, exc_info=True)
        print('Unable to extract: {}'.format(filename))
        print(e)
        global error_list
        error_list += [filename]
        
    return IDSemanticsColumns

# TEST function
logging.info('- TEST START -')

test_file = file_list[3]
print(test_file)
logging.info('filename: {}'.format(test_file))

test_extract = extractIDSemanticsWithColumnNames(test_file)
print (test_extract)
logging.info('extraction: {}'.format(test_extract))

print (error_list)
logging.info('error list: {}'.format(error_list))

logging.info('- TEST END -')