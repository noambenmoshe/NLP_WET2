from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
from sklearn.metrics import f1_score
from transformers import TrainingArguments
from transformers import Trainer
import torch
import argparse
import matplotlib.pyplot as plt
import random
from pathlib import Path

import numpy as np
import os
import json

try:
    import wandb
except:
    wandb = None

def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    f1 = f1_score(preds, labels, average='binary')
    acc = np.sum(preds == labels) / np.size(labels)
    return {'f1': f1, 'acc': acc}


def data_stats(raw_dataset, tokenizer, out_dir):
    train_len = []
    val_len = []

    tokenized_data = raw_dataset.map(tokenizer, input_columns='review')
    for i in tokenized_data['train']:
        train_len.append(len(i['input_ids']))

    for i in tokenized_data['test']:
        val_len.append(len(i['input_ids']))


    # plt.hist(train_len + val_len,bins=1000, cumulative=True, label='CDF',
    #          histtype='step', alpha=0.8, color='k')
    # plt.title('reviews lengths CDF')
    # plt.savefig(os.path.join(out_dir, 'review_length.png'))

    plt.figure()
    plt.hist(train_len,bins=1000, cumulative=True, label='CDF',
             histtype='step', alpha=0.8, color='k', density=True)
    plt.title('train reviews lengths CDF')
    plt.xlim(xmin=0, xmax=500)
    plt.savefig(os.path.join(out_dir, 'train_review_length.png'))
    plt.show()

    plt.figure()
    plt.hist(val_len, bins=1000, cumulative=True, label='CDF',
             histtype='step', alpha=0.8, color='k', density=True)
    plt.title('validation reviews lengths CDF')
    plt.xlim(xmin=0, xmax=500)

    plt.savefig(os.path.join(out_dir, 'val_review_length.png'))
    plt.show()

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Raw text dir", type=str, default='./HW2 - wet/clean_data')

    parser.add_argument("--model_name", type=str, default='bert-large-uncased')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=400)
    parser.add_argument("--out_dir", help="dir to save trained model", type=str, default='./trained_bert_large')
    args = parser.parse_args()
    return args

def main(args):

    if wandb:
        wandb.init(project="NLP-WET2", config=args)

    set_seed(args.seed)

    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                                                  num_labels=args.num_labels)

    DATA_PATH = Path(args.data_dir)
    data_files = {
        'train': str(DATA_PATH / 'baby/train.csv'),
        'test': str(DATA_PATH / 'baby/dev.csv')
    }
    raw_datasets = load_dataset("csv", data_files=data_files)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # data stats
    plots_dir = os.path.join(args.out_dir, 'plots/')
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    # data_stats(raw_datasets, tokenizer, out_dir=plots_dir)

    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', fn_kwargs={"max_length": args.max_length,
                                                                                        "truncation": True,
                                                                                        "padding": "max_length"})

    tokenized_datasets.set_format('torch')

    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    OUT_PATH = Path(args.out_dir)

    office_data_files = {
        'dev': str(DATA_PATH / 'office_products/dev.csv')
    }
    office_raw_datasets = load_dataset("csv", data_files=office_data_files)

    office_tokenized_datasets = office_raw_datasets.map(tokenizer, input_columns='review',
                                                        fn_kwargs={"max_length": args.max_length,
                                                                   "truncation": True,
                                                                   "padding": "max_length"})

    office_tokenized_datasets.set_format('torch')
    for split in office_tokenized_datasets:
        office_tokenized_datasets[split] = office_tokenized_datasets[split].add_column('label',
                                                                                       office_raw_datasets[split][
                                                                                           'label'])

    training_args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True,
                                      per_device_train_batch_size=args.batch_size,
                             per_device_eval_batch_size=args.batch_size,
                             gradient_accumulation_steps=args.grad_accum,
                             save_strategy='epoch',
                            save_total_limit=1,
                             load_best_model_at_end=True,
                             metric_for_best_model='acc',
                             greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=args.epochs,
                             report_to='wandb' if wandb else None,
                             )

    trainer = Trainer(
        model=model_seq_classification,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn
    )

    # # try train on baby , evaluate on office
    # trainer = Trainer(
    #         model=model_seq_classification,
    #         args=training_args,
    #         train_dataset=tokenized_datasets['train'],
    #         eval_dataset=office_tokenized_datasets['dev'],
    #         compute_metrics=metric_fn
    #     )

    trainer.train()
    if wandb:
        wandb.finish()
    # save best model
    save_model(model_seq_classification, vars(args), os.path.join(args.out_dir, 'best_model/'))

    print("====================================================")
    print("load and evaluate best model - baby")
    eval_model, _ = load_model(os.path.join(args.out_dir, 'best_model/'))
    # eval_trainer
    eval_args = TrainingArguments(output_dir=OUT_PATH,
                                  per_device_train_batch_size=args.batch_size,
                                  per_device_eval_batch_size=args.batch_size,
                                  gradient_accumulation_steps=args.grad_accum,
                                  metric_for_best_model='acc',
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

    print("====================================================")
    print("evaluate on office products")

    office_eval_trainer = Trainer(model=eval_model,
                           args=eval_args,
                           eval_dataset=office_tokenized_datasets['dev'],
                           compute_metrics=metric_fn
                           )

    eval_res = office_eval_trainer.evaluate()
    print("best model eval results: ", eval_res)

    return model_seq_classification

if __name__ == '__main__':
    args = get_args()
    main(args)