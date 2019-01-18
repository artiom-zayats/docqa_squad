import os
import glob
import json
import pdb
import numpy as np
from shutil import copyfile





pathold = os.path.expanduser('~/azayats/data/race/RACE_data/dev/combined')
pathnew = os.path.expanduser('~/azayats/data/fake_squad')
newfilename = 'dev-v1.1.json'
newfilepath = os.path.join(pathnew,newfilename)

if os.path.isfile(newfilepath):
    os.remove(newfilepath)

datafromfiles = {}
datanew = {'data': []}

#pdb.set_trace()
for filename in os.listdir(pathold):
    filepath = os.path.join(pathold, filename)

    
    
    with open(filepath) as json_file:  
        data = json.load(json_file)
        #for key, value in data.items():
        text = data['article']
        questions = data["questions"]
        answers = data["answers"]
        choices = data["options"]
        paragraph = {}
        paragraph = {'title' : filename}
        qas = []
        for i,question in enumerate(questions):
            qdic = {}

            qwords = question.split()
            if '_' in qwords:
                temp = ['xxemptytockenxx' if x == '_' else x for x in qwords]
                question = ' '.join(temp)
                #pdb.set_trace()

            qdic['question'] = question
            qdic['choices'] = choices[i]
            qdic['answer'] = answers[i]
            qdic['answers'] = [{'text': text.split()[0], 'answer': 0}] * 3
            qdic['id'] = filename[:-4]+'q'+str(i)
            qas.append(qdic)
        paragraph['paragraphs'] = [{'qas':qas,'context':text}]
    datanew['data'].append(paragraph)

with open(newfilepath, 'w') as outfile:
    json.dump(datanew , outfile)

#copyfile(newfilepath, os.path.join(pathnew,'train-v1.1.json'))

