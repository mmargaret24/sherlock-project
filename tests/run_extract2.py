DIR_DATASET = '/ivi/inde/mmargaret/data_search_e_data_csv/'
DIR_LOG = '/ivi/inde/mmargaret/sherlock-project/log/'
DIR_OUTPUT = '/ivi/inde/mmargaret/sherlock-project/output/'

import logging
from datetime import datetime
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='{}{}'.format(DIR_LOG,datetime.now().strftime('%Y%m%d_%H%M_sherlock.log')), mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)


# In[7]:


import numpy as np
import pandas as pd
import pyarrow as pa
import os
import sys
    
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

# In[ ]:


prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()


# ### Get list of all dataset

# In[ ]:


_ = os.listdir(DIR_DATASET)


# In[ ]:


file_list = [id for id in _ if '.csv' in id]
logging.info('Number of Files: {}'.format(len(file_list)))
print('Number of Files: {}'.format(len(file_list)))


# ## Extract features

# In[5]:


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
        logging.error('Unable to extract: {}'.format(filename))
        print('Unable to extract: {}'.format(filename))
        
        print(e)
        logging.error(e, exc_info=True)
        
        global error_list
        error_list += [filename]
        
    return IDSemanticsColumns


# In[27]:


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


# ### Get list of latest extracted semantics

# In[6]:


_ = os.listdir(DIR_OUTPUT)
_


# In[1]:


enrich_list = []

try:
    output_list = [DIR_OUTPUT + str(id) for id in _ if 'enriched_part_' in id]
    logging.info('Number of Output Files: {}'.format(len(output_list)))
    print('Number of Output Files: {}'.format(len(output_list)))

    latest_output = max(output_list, key=os.path.getctime)
    logging.info('Latest Output Filename: {}'.format(latest_output))
    print('Latest Output Filename: {}'.format(latest_output))

    output_df = pd.read_csv(latest_output)
    output_filenames = output_df['data_filename'].tolist()
    enrich_list = output_df.to_dict('records')
    
    logging.info('Number of Extracted Dataset: {}'.format(len(output_filenames)))
    print('Number of Extracted Dataset: {}'.format(len(output_filenames)))

except Exception as e:
    logging.error('Unable to retrieve latest output')
    print('Unable to retrieve latest output')
    
    logging.error(e, exc_info=True)
    print(e)
    pass

output_filenames[:5]


# ### Start Extraction

# In[ ]:


logging.info('- EXTRACT START -')

for i in range(0, len(file_list)):
    
    # so that it does not need to rerun existing output
    if (file_list[i] in output_filenames):
        logging.info('Existed: {} skipped'.format(file_list[i]))
        print('Existed: {} skipped'.format(file_list[i]))
        continue
        
    enrich_list += [extractIDSemanticsWithColumnNames(file_list[i])]
    if i%10==0:
        
        logging.info('i: {}'.format(i))
        sys.stdout.write('- i: {} -'.format(i))
        sys.stdout.write('\n')
        
        pd.DataFrame(enrich_list
                     , columns=['data_filename', 'colSemantics', 'colNames']).to_csv(DIR_OUTPUT +'enriched_part_' + str(i) +'.csv'
                     , index=False)
        
        pd.DataFrame(error_list
             , columns=['data_filename']).to_csv(DIR_OUTPUT + 'error_part_' + str(i) +'.csv'
             , index=False)
        
logging.info('- EXTRACT END -')
        


# In[ ]:


pd.DataFrame(enrich_list
             , columns=['data_filename', 'colSemantics', 'colNames']).to_csv(DIR_OUTPUT +'enriched_all.csv'
             , index=False)


# In[ ]:


pd.DataFrame(error_list
             , columns=['data_filename']).to_csv(DIR_OUTPUT + 'error_all.csv'
             , index=False)

