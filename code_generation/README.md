# Code-Generation

## Task Definition

The task is to generate code snippet for a given natural language comments, and evaluted by [Pass@k](https://arxiv.org/abs/2107.03374) score.


### Dependency and setup

You will need Python 3.8 or higher.

### Zero-shot setting

For Zero-shot setting, you can follow the [MultiPL-E](https://github.com/nuprl/MultiPL-E/tree/main) repo. Using the following script:
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
For different model, just replace [SantaCoder](https://huggingface.co/bigcode/gpt_bigcode-santacoder) model with the model name.
### Few-shot setting

For Few-shot setting, please refer to the one.txt to ten.txt for simple R demo. Follow the same apporch as zero-shoting but use the following script:
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

 Place the sample R code after prefix--prompt-prefix.

 For BM25 or embedding few-shot, replace the completions.py located under [MultiPL-E/multipl_e/](https://github.com/nuprl/MultiPL-E/tree/main/multipl_e) with completion_few_shot.py provided here and follow the following script:
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
 Where prompt num is the number of example you want to include in your few-shot examples.