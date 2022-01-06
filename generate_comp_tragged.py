import argparse
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import numpy as np
import os
import json
from main import load_model, metric_fn
import shutil
import csv

def create_labeled_csv(predicts,original_file, dest_file):
    # shutil.copyfile(original_file, dest_file)

    with open(original_file, 'r') as csvinput:
        with open(dest_file, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            # row = next(reader)
            # row.append('Berry')
            # all.append(row)

            for i,row in enumerate(reader):
                if i!=0:
                    row.append(True if predicts[i-1] else False)
                all.append(row)

            writer.writerows(all)
            assert len(predicts)+1  == len(all), f' len(predicts)={len(predicts)} len(all)={len(all)} expected to have a prediction to every example in input file'

    print('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Raw text dir", type=str, default='./HW2 - wet/clean_data/office_products')
    parser.add_argument("--result_filename", help="Name of the results file", type=str, default='comp_313627358.csv')
    parser.add_argument("--out_dir", help="dir for predict results", type=str, default='./predict')
    parser.add_argument("--model_dir", help="dir of trained model", type=str, default='./best_model')
    args = parser.parse_args()

    DATA_PATH = Path(args.data_dir)
    print("organize data")
    if 'baby' in args.data_dir:
        domain = 'baby'
        data_files = {
            'unlabeled': str(DATA_PATH / 'unlabeled.csv')
        }
        data_names = ['unlabeled']

    if 'office' in args.data_dir:
        domain = 'office'
        data_files = {
            'test': str(DATA_PATH / 'test.csv'),
            'unlabeled': str(DATA_PATH / 'unlabeled.csv')
        }
        data_names = ['test','unlabeled']

    raw_datasets = load_dataset("csv", data_files=data_files)

    print("load trained model")
    eval_model, config = load_model(args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', fn_kwargs={"max_length": config['max_length'],
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"})

    tokenized_datasets.set_format('torch')

    # eval_trainer
    eval_args = TrainingArguments(output_dir=args.out_dir,
                                  per_device_train_batch_size=config['batch_size'],
                                  per_device_eval_batch_size=config['batch_size'],
                                  gradient_accumulation_steps=config['grad_accum'],
                                  metric_for_best_model='f1',
                                  greater_is_better=True, evaluation_strategy='epoch', do_train=False,
                                  do_eval=True,
                                  num_train_epochs=config['epochs'], report_to='none',
                                  )




    eval_trainer = Trainer(model=eval_model,
                           args=eval_args,
                           compute_metrics=metric_fn
                           )




    predict_res = eval_trainer.predict(tokenized_datasets[data_names[0]])
    print(f'domain = {domain}')
    preds = np.argmax(predict_res.predictions, axis=1)
    print("best model predict results: ", preds)

    create_labeled_csv(preds,data_files[data_names[0]],args.result_filename)

if __name__ == '__main__':
    main()