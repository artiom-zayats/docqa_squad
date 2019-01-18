import os
import glob
import json
import pdb
import numpy



path = os.path.expanduser('~/azayats/data/race_test_data/temp')

for filename in os.listdir(path):
    newfilename = 'new'+filename

    filepath = os.path.join(path, filename)
    #numpy.random.seed(0)


    ans_array = ['A','B','C','D']
    
    
    with open(filepath) as json_file:  
        data = json.load(json_file)
        #data['data'][0]['paragraphs'][0]['qas'][0]['choices'] = ['A','AA','AAA','AAAA']
        for d in range(len(data['data'])):
            for p in range(len(data['data'][d]['paragraphs'])):
                for q in range(len(data['data'][d]['paragraphs'][p]['qas'])):
                    data['data'][d]['paragraphs'][p]['qas'][q]['choices'] = ['A A','AA AA AA','AAA AAA AAA AAA AAA','AAAA AAAA AAAA AAAA AAAA AAAA']
                    random_index = numpy.random.randint(0,len(ans_array))
                    data['data'][d]['paragraphs'][p]['qas'][q]['answer'] = ans_array[random_index]

        with open(os.path.join(path,newfilename), 'w') as outfile:
            json.dump(data, outfile)
