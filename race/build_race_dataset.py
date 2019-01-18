import argparse
import json
import urllib
from os import listdir, mkdir
from os.path import expanduser, join, exists
from typing import List

#import code
import pdb

from tqdm import tqdm

from docqa import config
from docqa.race.race_data import Question, Document, Paragraph, SquadCorpus
from docqa.data_processing.span_data import ParagraphSpan, ParagraphSpans
from docqa.data_processing.text_utils import get_word_span, space_re, NltkAndPunctTokenizer
from docqa.utils import flatten_iterable

"""
Script to build a corpus from RACE training data 
"""


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undoes it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def parse_race_data(source, name, tokenizer, use_tqdm=True) -> List[Document]:
    with open(source, 'r') as f:
        source_data = json.load(f)
    
    if use_tqdm:
        iter_files = tqdm(source_data['data'], ncols=80)
    else:
        iter_files = source_data['data']

    mult_options_dict = {'A':0,'B':1,'C':2,'D':3}
    mult_anwsers_array = []

    for article_ix, article in enumerate(iter_files):
        article_ix = "%s-%d" % (name, article_ix)

        paragraphs = []

        for para_ix, para in enumerate(article['paragraphs']):
            questions = []
            #pdb.set_trace()
            context = para['context']

            tokenized = tokenizer.tokenize_with_inverse(context)
            # list of sentences + mapping from words -> original text index
            text, text_spans = tokenized.text, tokenized.spans
            flat_text = flatten_iterable(text)

            n_words = sum(len(sentence) for sentence in text)

            for question_ix, question in enumerate(para['qas']):
                # There are actually some multi-sentence questions, so we should have used
                # tokenizer.tokenize_paragraph_flat here which would have produced slighy better
                # results in a few cases. However all the results we report were
                # done using `tokenize_sentence` so I am just going to leave this way
                question_text = tokenizer.tokenize_sentence(question['question'])
                #vz
                #pdb.set_trace()
                choices_text = [tokenizer.tokenize_sentence(x) for x in question['choices']]
                mult_answer = question['answer']
                mult_answer_ix = mult_options_dict[mult_answer]
                
                #old calc of spans from 'build_squad_dataset'

                mult_anwsers_array.append(mult_answer_ix)
                
                questions.append(Question(question['id'], question_text, mult_answer_ix,choices_text))

            paragraphs.append(Paragraph(text, questions, article_ix, para_ix, context,mult_anwsers_array))
            #vz we need to add here choices as well (like questions or similar)

        yield Document(article_ix, article["title"], paragraphs)


def main():
    parser = argparse.ArgumentParser("Preprocess RACE data")
    #basedir = join(expanduser("~"), "data", "squad")
    basedir = join(expanduser("~"), "azayats", "data", "fake_squad")
    parser.add_argument("--train_file", default=join(basedir, "train-v1.1.json"))
    parser.add_argument("--dev_file", default=join(basedir, "dev-v1.1.json"))

    if not exists(config.CORPUS_DIR):
        mkdir(config.CORPUS_DIR)

    target_dir = join(config.CORPUS_DIR, SquadCorpus.NAME)
    if exists(target_dir) and len(listdir(target_dir)) > 0:
        raise ValueError("Files already exist in " + target_dir)

    args = parser.parse_args()
    tokenzier = NltkAndPunctTokenizer()

    print("Parsing train...")
    train = list(parse_race_data(args.train_file, "train", tokenzier))

    print("Parsing dev...")
    dev = list(parse_race_data(args.dev_file, "dev", tokenzier))

    #print("Parsing test...")
    #dev = list(parse_race_data(args.test_file, "test", tokenzier))

    print("Saving...")
    SquadCorpus.make_corpus(train, dev)
    print("Done")


if __name__ == "__main__":
    main()
