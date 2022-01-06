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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Raw text dir", type=str, default='./HW2 - wet/clean_data/baby')
    parser.add_argument("--result_filename", help="Name of the results file", type=str, default='comp_313627358.csv')
    parser.add_argument("--out_dir", help="dir for predict results", type=str, default='./predict')
    parser.add_argument("--model_dir", help="dir of trained model", type=str, default='./best_model')
    args = parser.parse_args()

    DATA_PATH = Path(args.data_dir)
    print("organize data")
    if 'baby' in args.data_dir:
        domain = 'baby'
        data_files = {
            'train': str(DATA_PATH / 'train.csv'),
            'test': str(DATA_PATH / 'dev.csv')
        }
    if 'office' in args.data_dir:
        domain = 'office'
        data_files = {
            'dev': str(DATA_PATH / 'dev.csv'),
            'test': str(DATA_PATH / 'test.csv')
        }

    raw_datasets = load_dataset("csv", data_files=data_files)

    print("load trained model")
    eval_model, config = load_model(args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', fn_kwargs={"max_length": config['max_length'],
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"})

    tokenized_datasets.set_format('torch')

    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    # eval_trainer
    eval_args = TrainingArguments(output_dir=args.out_dir,
                                  per_device_train_batch_size=config['batch_size'],
                                  per_device_eval_batch_size=config['batch_size'],
                                  gradient_accumulation_steps=config['grad_accum'],
                                  metric_for_best_model='f1',
                                  greater_is_better=True, evaluation_strategy='epoch', do_train=False,
                                  do_eval=True,
                                  num_train_epochs=args.epochs, report_to='none',
                                  )




    eval_trainer = Trainer(model=eval_model,
                           args=eval_args,
                           train_dataset=tokenized_datasets['train'] if domain == 'baby' else None,
                           eval_dataset=tokenized_datasets['test'] if domain == 'baby' else tokenized_datasets['dev'],
                           compute_metrics=metric_fn
                           )

    eval_res = eval_trainer.evaluate()
    print("best model eval results: ", eval_res)

    predict_res = eval_trainer.predict(tokenized_datasets['test'])
    print("best model predict results: ", predict_res)

    if domain == 'baby':
        predict_res = eval_trainer.predict(tokenized_datasets['train'])
    if domain == 'office':
        predict_res = eval_trainer.predict(tokenized_datasets['dev'])
    print(f'domain = {domain}')
    print("best model predict results: ", predict_res)

if __name__ == '__main__':
    main()