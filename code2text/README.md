# Code-To-Text

## Task Definition

The task is to generate natural language comments for a given code snippet, and evaluted by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Monolingual fine-tuning and inference

For monolingual fine-tuning and inference, please clone the "CodeXGLUE" [repo](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) and follow the instruction.

## Multilingual fine-tuning and inference

For multilingual fine-tuning and inference, please download the dataset from [here](https://zenodo.org/records/5683528)

## R language

For monolingual fine-tuning and inference, please download the dataset from [here](https://zenodo.org/records/13871742)


### Dependency

- python 3.8
- torch==1.12.0
- transformers==4.20.1

### Fine-tune

To fine-tune encoder-decoder CodeBERT on the dataset, for GraphCodeBERT and other models, follow the exact instructions followed for CodeBERT, just replace the "microsoft/codebert-base" with the targeet model.

```shell
cd code
lang=ruby #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
```


### Inference

```shell
batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

### Evaluation

```shell
python ../evaluator/evaluator.py model/$lang/test_1.gold < model/$lang/test_1.output
```


