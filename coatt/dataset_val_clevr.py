import os
import operator
import numpy as np
import json

from six.moves import cPickle as pickle
from collections import defaultdict


image_dir = "../../CLEVR_v1.0/images/val"
img_prefix = "CLEVR_val_"
qjson = "../../CLEVR_v1.0/questions/CLEVR_val_questions.json"

with open('../data_clevr/a2i.pkl', 'rb') as f:
    a2i = pickle.load(f)

with open(qjson, 'r') as f:
    dataset = json.load(f)

ques_ids = []
qa = []
qqa = []
for ques_id in dataset['questions']:
    ques_ids.append(ques_id['question_index'])
    qa.append(ques_id['answer'])
    qqa.append(ques_id['question'])


img_names = [f for f in os.listdir(image_dir) if '.png' in f]
img_ids = []
for fname in img_names:
    img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
    img_ids.append(int(img_id))

ques_ids_modif = []
for ques_id in ques_ids:
    ques = qqa[ques_id]
    ans = qa[ques_id]
    answer = ""
    ans = ans.lower()
    if ans in a2i:
       answer = ans
       break

    if not answer == "":
        ques_ids_modif.append(ques_id)

np.save('../data_clevr/va_img_names.npy', img_names)
np.save('../data_clevr/va_img_ids.npy', img_ids)
np.save('../data_clevr/va_ques_ids.npy', ques_ids_modif)
