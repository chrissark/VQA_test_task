import os
import operator
import numpy as np

from collections import defaultdict
from external.vqa.vqa import VQA

def pre_process_dataset(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if '.png' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    ques_ids = vqa.getQuesIds(img_ids)

    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    start = q2i["<sos>"]
    end = q2i["<eos>"]
    UNK = q2i["<unk>"]

    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        qqa = vqa.loadQQA(ques_id)[0]

        ques = qqa['question'][:-1]
        [q2i[x] for x in ques.lower().strip().split(" ")]

        answers = qa['answers']
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
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

    return q2i, a2i, i2a, a2i_count

if __name__ == '__main__':
    image_dir = "C:/Users/Admin/Documents/9_sem/VQA/Images/abstract_v002/train2015"
    img_prefix = "abstract_v002_train2015_"
    qjson = "C:/Users/Admin/Documents/9_sem/VQA/Questions/OpenEnded_abstract_v002_train2015_questions.json"
    ajson = "C:/Users/Admin/Documents/9_sem/VQA/Annotations/abstract_v002_train2015_annotations.json"

    q2i, a2i, i2a, a2i_count = pre_process_dataset(image_dir, qjson, ajson, img_prefix)
    np.save('./data/q2i.npy', q2i)
    np.save('./data/a2i.npy', a2i)
    np.save('./data/i2a.npy', i2a)
    np.save('./data/a2i_count.npy', a2i_count)


