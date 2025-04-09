# Code-Generation

## Task Definition

The task is to generate code snippet for a given natural language comments, and evaluted by [Pass@k](https://arxiv.org/abs/2107.03374) score.


### Dependency and setup

1. You will need Python 3.8 or higher.

2. You will need to install some Python packages:

    ```bash
    pip3 install aiohttp numpy tqdm pytest datasets torch transformers
    ```

3. You need to install one of [Podman] or [Docker].

3. Check out the repository:    

   ```bash
   git clone https://github.com/nuprl/MultiPL-E
   ```

4. Enter the repository directory:

   ```bash
   cd MultiPL-E
   ```

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


