import argparse
import json
from typing import List

import numpy as np
import pdb

from docqa import trainer
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextAndQuestion ,ContextLenKey

from docqa.dataset import FixedOrderBatcher , ClusteredBatcher
from docqa.evaluator import Evaluator, Evaluation, SpanEvaluator , MultiChoiceEvaluator
from docqa.model_dir import ModelDir
from docqa.race.race_data import SquadCorpus, split_docs
from docqa.utils import transpose_lists, print_table

"""
Run an evaluation on squad and record the official output
"""


class RecordSpanPrediction(Evaluator):
    def __init__(self, bound: int):
        self.bound = bound

    def tensors_needed(self, prediction):
        #span, score = prediction.get_best_span(self.bound)
        
        span , score = prediction.get_prediction()
        return dict(spans = span, model_scores = score)
        #pdb.set_trace()
        #score = 0
        #return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        spans, model_scores = kargs["spans"], kargs["model_scores"]
        results = {"model_conf": model_scores,
                   "predicted_span": spans,
                   "question_id": [x.question_id for x in data]}
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on SQuAD')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument("-o", "--official_output", type=str, help="where to output an official result file")
    parser.add_argument('-n', '--sample_questions', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[17],
                        help="Max size of answer")
    parser.add_argument('-b', '--batch_size', type=int, default=45,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-s', '--step', default=None,
                        help="Weights to load, can be a checkpoint step or 'latest'")
    parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    args = parser.parse_args()

    num_choices = 4

    model_dir = ModelDir(args.model)

    corpus = SquadCorpus()
    if args.corpus == "dev":
        questions = corpus.get_dev()
    else:
        questions = corpus.get_train()
    questions = split_docs(questions)

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]


    questions.sort(key=lambda x:x.n_context_words, reverse=True)
    #pdb.set_trace()
    #print(args.batch_size)
    #dataset = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, False),None,num_choices)


    dataset = ParagraphAndQuestionDataset(questions, ClusteredBatcher(45, ContextLenKey(), False, False),None,num_choices)
    
    #ClusteredBatcher(45, ContextLenKey(), False, False)

    evaluators = [MultiChoiceEvaluator(num_choices)]
    #if args.official_output is not None:
        #evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))
    #pdb.set_trace()
    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    model = model_dir.get_model()
    #pdb.set_trace()
    evaluation = trainer.test(model, evaluators, {args.corpus: dataset},
                              corpus.get_resource_loader(), checkpoint, not args.no_ema)[args.corpus]
    
    #pdb.set_trace()
    
    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))

    # Save the official output
    if args.official_output is not None:
        data_to_dump = {}

        list_of_choices = ['A','B','C','D']

        q_ids = evaluation.per_sample["question_id"]
        correct_ans = evaluation.per_sample["correct answer"]
        correct_ids = evaluation.per_sample["correct index"]
        pred_ids = evaluation.per_sample["predictied index"]
        pred_ans = evaluation.per_sample["predictied answer"]
        is_correct  = evaluation.per_sample["is correct"]
        #pdb.set_trace()
        for ix, q_ids in enumerate(q_ids):
            if(is_correct[ix]):
                data_to_dump[q_ids] = {'Is Correct' : 'True',
                 'predictied' : [' '.join(pred_ans[ix]),list_of_choices[pred_ids[ix]]],
                 'correct' : [' '.join(correct_ans[ix]),list_of_choices[correct_ids[ix]]]
                } 
            else:
                data_to_dump[q_ids] = {'Is Correct' : 'False',
                 'predictied' : [' '.join(pred_ans[ix]),list_of_choices[pred_ids[ix]]],
                 'correct' : [' '.join(correct_ans[ix]),list_of_choices[correct_ids[ix]]]
                } 
        #pdb.set_trace()
        with open(args.official_output, "w") as f:
            json.dump(data_to_dump , f)


if __name__ == "__main__":
    main()
    # tmp()



