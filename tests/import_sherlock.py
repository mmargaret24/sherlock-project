import os
_ = os.listdir('/ivi/inde/mmargaret/data_search_e_data_csv/')
file_list = [id for id in _ if '.csv' in id]
len(file_list)

import numpy as np
import pandas as pd
import pyarrow as pa

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

prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()