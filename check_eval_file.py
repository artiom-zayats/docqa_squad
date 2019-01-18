import os
import glob
import json
import pdb
import numpy



path = os.path.expanduser('~/azayats/projects/document-qa_edited/')
filename = 'test_output_1.json'
filepath = os.path.join(path, filename)

bol_pred = []

with open(filepath) as json_file:  
        data = json.load(json_file)
        for ix,key in enumerate(data.keys()):
            if data[key]['Is Correct'] == 'False' :
                bol_pred.append(0)
            else:
                bol_pred.append(1)
acc = sum(bol_pred)/len(bol_pred)
print(acc)
