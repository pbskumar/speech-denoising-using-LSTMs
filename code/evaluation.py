
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io.wavfile as sio
import mir_eval
import matplotlib.pyplot as plt
import matlab.engine
import glob
import pickle
import pprint


# In[2]:

eng = matlab.engine.start_matlab()

predicted_file_names = glob.glob('test_results/*/*/*predicted.wav', recursive=True)
original_file_names = glob.glob('test_results/*/*/*original.wav', recursive=True)


# In[4]:

for i in range(len(predicted_file_names)):
    assert predicted_file_names[i].split('\\')[:-1] == original_file_names[i].split('\\')[:-1]


# In[7]:

models = set([f_name.split('\\')[1] for f_name in original_file_names])
data_sets = set([f_name.split('\\')[2] for f_name in original_file_names])
scores = ['sdr', 'stoi', 'count']


# In[9]:

def create_scores_dict(models_list, datasets_list, scores_list):
    _score_dict = {}
    
    for model in models_list:
        model_temp_dict = {}
        for dataset in datasets_list:
            
            data_temp_dict = {}
            for score in scores_list:
                data_temp_dict[score] = 0
            
            model_temp_dict[dataset] = data_temp_dict
        _score_dict[model] = model_temp_dict
    
    return _score_dict


# In[10]:

scores_dict = create_scores_dict(models, data_sets, scores)


# In[12]:

f_len = len(original_file_names)
for i in range(f_len):
    
    print('Processing file: {a:>6d} of {b:<6d}'.format(a=i+1, b=f_len))
    
    # splitting filename
    file_name_split = original_file_names[i].split('\\')
    
    # reading data
    _, s = sio.read(filename=original_file_names[i])
    _, e = sio.read(filename=predicted_file_names[i])    
    
    # initializing parameters
    model_name = file_name_split[1]
    data_set = file_name_split[2]
    
    model_dict = scores_dict[model_name]
    dataset_dict = model_dict[data_set]
    
    # calculating scores
    sdr, _, _, _ = mir_eval.separation.bss_eval_sources(s, e)
    stoi = eng.stoi(matlab.double(s.tolist()), matlab.double(e.tolist()))
    
    
    
    # updating the score fields
    dataset_dict['count'] += 1
    dataset_dict['sdr'] += sdr
    dataset_dict['stoi'] += stoi
        


# In[13]:

for model in scores_dict:
    model_dict = scores_dict[model]
    for data_set in model_dict:
        data_set_dict = model_dict[data_set]
        
        data_set_dict['sdr'] = data_set_dict['sdr'] / data_set_dict['count']
        data_set_dict['stoi'] = data_set_dict['stoi'] / data_set_dict['count']


# In[15]:

pickle.dump(obj=scores_dict, file=open('model_evals.pkl', 'wb'))


# In[16]:

eng.quit()


# In[ ]:



