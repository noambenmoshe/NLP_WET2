from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import TrainingArguments
from transformers import Trainer
import torch
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import json

def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'f1': f1_score(preds, labels, average='binary')}


def data_stats(raw_dataset, tokenizer):
    train_len = []
    val_len = []

    tokenized_data = raw_dataset.map(tokenizer, input_columns='review')
    for i in tokenized_data['train']:
        train_len.append(len(i['input_ids']))

    for i in tokenized_data['test']:
        val_len.append(len(i['input_ids']))

    #
    # plt.hist(train_len,bins=1000, cumulative=True, label='CDF',
    #          histtype='step', alpha=0.8, color='k')
    # plt.title('train sentences lengths CDF')
    # plt.show()
    #
    # plt.hist(val_len, bins=1000, cumulative=True, label='CDF',
    #          histtype='step', alpha=0.8, color='k')
    # plt.title('validation sentences lengths CDF')
    # plt.show()
    return

def load_model(dir):
    with open(os.path.join(dir,'config.json'), "r") as fp:
        config = json.load(fp)

    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'],
                                                                                  num_labels=config['num_labels'])
    model.load_state_dict(torch.load(os.path.join(dir, 'model.pt')))
    return model, config


def save_model(model, config, dir):
    '''
    saves model state_dict and config dictionary for reloading
    :param model: model to save
    :param config: moedl configuration params
    :param dir: where to save the model
    :return: None
    '''
    Path(dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.to('cpu').state_dict(), os.path.join(dir, 'model.pt'))
    with open(os.path.join(dir,'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Raw text dir", type=str, default='./HW2 - wet/clean_data/baby')

    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=350)
    parser.add_argument("--out_dir", help="dir to save trained model", type=str, default='./trained_mlp')
    args = parser.parse_args()

    set_seed(args.seed)

    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                                                  num_labels=args.num_labels)

    DATA_PATH = Path(args.data_dir)
    data_files = {
        'train': str(DATA_PATH / 'train.csv'),
        'test': str(DATA_PATH / 'dev.csv')
    }
    raw_datasets = load_dataset("csv", data_files=data_files)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_stats(raw_datasets, tokenizer)
    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', fn_kwargs={"max_length": args.max_length,
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"})

    tokenized_datasets.set_format('torch')

    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    OUT_PATH = Path(args.out_dir)

    training_args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=args.batch_size,
                             per_device_eval_batch_size=args.batch_size,
                             gradient_accumulation_steps=args.grad_accum,
                             save_strategy='epoch',
                             save_total_limit=1,  # Only last 5 models are saved. Older ones are deleted.
                             load_best_model_at_end=True,
                             metric_for_best_model='f1',
                             greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=args.epochs, report_to='none',
                             )

    trainer = Trainer(
        model=model_seq_classification,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn
    )

    trainer.train()

    # save best model
    save_model(model_seq_classification, vars(args), os.path.join(args.out_dir, 'best_model/'))

    print("====================================================")
    print("load and evaluate best model")
    eval_model, _ = load_model(os.path.join(args.out_dir, 'best_model/'))
    # eval_trainer
    eval_args = TrainingArguments(output_dir=OUT_PATH,
                                  per_device_train_batch_size=args.batch_size,
                                  per_device_eval_batch_size=args.batch_size,
                                  gradient_accumulation_steps=args.grad_accum,
                                  metric_for_best_model='f1',
                                  greater_is_better=True, evaluation_strategy='epoch', do_train=False,
                                  do_eval=True,
                                  num_train_epochs=args.epochs, report_to='none',
                                  )

    OUT_PATH = Path(os.path.join(args.out_dir, 'best_moedl_eval/'))
    eval_trainer = Trainer(model=eval_model,
                            args=eval_args,
                            train_dataset=tokenized_datasets['train'],
                            eval_dataset=tokenized_datasets['test'],
                            compute_metrics=metric_fn
                        )

    eval_res = eval_trainer.evaluate()
    print("best model eval results: ", eval_res)

