from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import TrainingArguments
from transformers import Trainer
import torch


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'f1': f1_score(preds, labels, average='binary')}


if __name__ == '__main__':

    model_name = 'bert-base-uncased'

    model_seq_classification = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    DATA_PATH = Path("./HW2 - wet/clean_data/baby")
    data_files = {
        'train': str(DATA_PATH / 'train.csv'),
        'test': str(DATA_PATH / 'dev.csv')
    }
    raw_datasets = load_dataset("csv", data_files=data_files)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review', fn_kwargs={"max_length": 128, "truncation": True, "padding": "max_length"})
    tokenized_datasets.set_format('torch')

    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    OUT_PATH = Path("./results")

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=64,
                             per_device_eval_batch_size=128, save_strategy='no', metric_for_best_model='dev_f1',
                             greater_is_better=True, evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=5, report_to='none')

    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn
    )

    trainer.train()