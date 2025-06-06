# Code-Generation

## Task Definition

The task is to generate code snippet for a given natural language comments, and evaluted by [Pass@k](https://arxiv.org/abs/2107.03374) score.


### Dependency and setup

You will need Python 3.8 or higher.

### Using Humaneval to benchmark the models for Python
To test the model's performance on Humaneval in Python, you can follow the [bigcode-evaluation-harness](https://github.com/zixiaozhao/bigcode-evaluation-harness) repo. Using the following script:
```
accelerate launch  main.py \
  --model model_path \
  --max_length_generation 512 \
  --tasks humaneval \
  --temperature 0.8 \
  --n_samples 100 \
  --batch_size 16 \
  --allow_code_execution
```

If starcoder2 is used as the test model, running the above command should give you pass@1: 29.23%, pass@10: 64.69. Note if you want to compute pass@5, change line 50 in bigcode_eval/tasks/humaneval.py so that K= [1, 5, 10]

### Zero-shot setting

For the Zero-shot setting, you can follow the [MultiPL-E](https://github.com/zixiaozhao/MultiPL-E) repo. Using the following script:
 ```
mkdir tutorial
python3 automodel.py \
    --name bigcode/gpt_bigcode-santacoder \
    --root-dataset humaneval \
    --lang r \
    --temperature 0.2 \
    --batch-size 20 \
    --completion-limit 20 \
    --output-dir-prefix tutorial
 ```
For a different model, just replace [SantaCoder](https://huggingface.co/bigcode/gpt_bigcode-santacoder) model with the model name. Note here the temperature is set to 0.2, for the greedy approach, you should not use this parameter.

Successfully running the script should give you pass@1 = 20.5 for starcoder2.

### Few-shot setting

For the Few-shot setting, please refer to the one.txt to ten.txt for a simple R demo. Follow the same approach as zero-shotting, but use the following script:
 ```
mkdir tutorial
python3 automodel.py \
    --prompt-prefix \
    --name bigcode/gpt_bigcode-santacoder \
    --root-dataset humaneval \
    --lang r \
    --temperature 0.2 \
    --batch-size 20 \
    --completion-limit 20 \
    --output-dir-prefix tutorial
 ```

 Place the sample R code after the prefix--prompt-prefix.

 For BM25 or embedding few-shot, replace the completions.py located under [MultiPL-E/multipl_e/](https://github.com/zixiaozhao/MultiPL-E/tree/main/multipl_e) with completion_few_shot.py provided here and follow the following script:
  ```
mkdir tutorial
python3 automodel.py \
    --prompt-num 1\
    --name bigcode/gpt_bigcode-santacoder \
    --root-dataset humaneval \
    --lang r \
    --temperature 0.2 \
    --batch-size 20 \
    --completion-limit 20 \
    --output-dir-prefix tutorial
 ```
 Where prompt num is the number of examples you want to include in your few-shot examples.
