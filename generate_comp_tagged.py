import argparse
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import numpy as np
from main import load_model, metric_fn
from main import main as train_model
import csv

def create_labeled_csv(predicts,original_file, dest_file):
    with open(original_file, 'r') as csvinput:
        with open(dest_file, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            for i,row in enumerate(reader):
                if i!=0:
                    row.append(True if predicts[i-1] else False)
                all.append(row)

            writer.writerows(all)
            assert len(predicts)+1  == len(all), f' len(predicts)={len(predicts)} len(all)={len(all)} expected to have a prediction to every example in input file'

    print('done')

def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_dir", help="Raw text dir", type=str, default='./clean_data/')
    parser.add_argument("--eval_domain", help="domain to test", type=str, default='office_products')
    parser.add_argument("--result_filename", help="Name of the results file", type=str, default='comp_313627358.csv')
    parser.add_argument("--out_dir", help="dir for predict results", type=str, default='./predict')
    parser.add_argument("--model_dir", help="dir of trained model", type=str, default=None)

    # train model args
    parser.add_argument("--model_name", type=str, default='bert-large-uncased')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=400)
    args = parser.parse_args()
    return args



def main():
    args = get_args()

    # load raw datasets
    DATA_PATH = Path(args.data_dir)
    print("organize data")
    if args.eval_domain == 'baby':
        domain = 'baby'
        data_files = {
            'unlabeled': str(DATA_PATH / 'baby/unlabeled.csv')
        }
        data_names = ['unlabeled']

    if args.eval_domain == 'office_products':
        domain = 'office'
        data_files = {
            'test': str(DATA_PATH / 'office_products/test.csv'),
            'unlabeled': str(DATA_PATH / 'office_products/unlabeled.csv')
        }
        data_names = ['test','unlabeled']

    raw_datasets = load_dataset("csv", data_files=data_files)

    # get trained model to test
    if args.model_dir is not None:
        print("load fine tuned model")
        eval_model, config = load_model(args.model_dir)

    else:
        print("train model [fine tuning]")
        eval_model = train_model(args)
        config = vars(args)

    # tokenize test dataset
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

    # generate test results
    predict_res = eval_trainer.predict(tokenized_datasets[data_names[0]])
    print(f'domain = {domain}')
    preds = np.argmax(predict_res.predictions, axis=1)
    print("best model predict results: ", preds)

    create_labeled_csv(preds,data_files[data_names[0]],args.result_filename)

if __name__ == '__main__':
    main()