# Code-To-Text

## Task Definition

The task is to generate natural language comments for a given code snippet, and evaluated by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Dataset structure
```
.
├── code/ # Evaluation & training scripts
│ ├── bleu.py # BLEU score computation
│ ├── model.py # Model definition / loading
│ └── run.py # Train / inference entry point
├── dataset/ # Multilingual data (parallel to code/)
│ ├── python
│ ├── java
│ ├── R
│ ├── ruby
│ └── ... # Additional languages (Go, JS, etc.)
```
Under each language, there should be three files: train.jsonl, test.jsonl and valid.jsonl, they are used for training, testing, and validation, respectively.

## Monolingual fine-tuning and inference

For monolingual fine-tuning and inference, please clone the "CodeXGLUE" [repo](https://github.com/zixiaozhao/CodeXGLUE/tree/main/Code-Text/code-to-text) and follow the instructions.

## Multilingual fine-tuning and inference

For multilingual fine-tuning and inference, please download the dataset from [here](https://zenodo.org/records/5683528)

## R language

For monolingual fine-tuning and inference, please download the dataset from [here](https://zenodo.org/records/13871742)


### Dependency

- python 3.8
- torch==1.12.1
- transformers==4.20.0

### Fine-tune

To fine-tune encoder-decoder CodeBERT on the dataset, for GraphCodeBERT and other models, follow the exact instructions followed for CodeBERT, just replace the "microsoft/codebert-base" with the target model.

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
For method name prediction, change the target_length to 10, and everything else remains the same.

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


