import os
import operator
import numpy as np
import json

from six.moves import cPickle as pickle
from collections import defaultdict

image_dir = "../../CLEVR_v1.0/images/train"
img_prefix = "CLEVR_train_"
qjson = "../../CLEVR_v1.0/questions/CLEVR_train_questions.json"

with open(qjson, 'r') as f:
    dataset = json.load(f)

img_names = [f for f in os.listdir(image_dir) if '.png' in f]
img_ids = []
for fname in img_names:
    img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
    img_ids.append(int(img_id))

ques_ids = []
qa = []
qqa = []
for ques_id in dataset['questions']:
    ques_ids.append(ques_id['question_index'])
    qa.append(ques_id['answer'])
    qqa.append(ques_id['question'])

q2i = defaultdict(lambda: len(q2i))
pad = q2i["<pad>"]
start = q2i["<sos>"]
end = q2i["<eos>"]
UNK = q2i["<unk>"]

a2i_count = {}
for ques_id in ques_ids:

    ques = qqa[ques_id]
    [q2i[x] for x in ques.lower().strip().split(" ")]

    ans = qa[ques_id]
    ans = ans.lower()
    if ans not in a2i_count:
        a2i_count[ans] = 1
    else:
        a2i_count[ans] = a2i_count[ans] + 1

a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)

i2a = {}
count = 0
a2i = defaultdict(lambda: len(a2i))
for word, _ in a_sort:
    a2i[word]
    i2a[a2i[word]] = word
    count = count + 1
    if count == 1000:
        break


with open('../data_clevr/q2i.pkl', 'wb') as f:
    pickle.dump(dict(q2i), f)
with open('../data_clevr/a2i.pkl', 'wb') as f:
    pickle.dump(dict(a2i), f)
with open('../data_clevr/i2a.pkl', 'wb') as f:
    pickle.dump(i2a, f)
with open('../data_clevr/a2i_count.pkl', 'wb') as f:
    pickle.dump(a2i_count, f)

np.save('../data_clevr/q2i.npy', dict(q2i))
np.save('../data_clevr/a2i.npy', dict(a2i))
np.save('../data_clevr/i2a.npy', i2a)
np.save('../data_clevr/a2i_count.npy', a2i_count)

np.save('../data_clevr/tr_img_names.npy', img_names)
np.save('../data_clevr/tr_img_ids.npy', img_ids)
np.save('../data_clevr/tr_ques_ids.npy', ques_ids)