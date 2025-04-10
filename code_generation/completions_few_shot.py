import datasets
import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
import logging
import voyageai
import time

voyageai.api_key = "pa-DmnodISyaBsp_BmM7qrt3Ad_xMF5EwxTNPxI7JRUYnL"
vo = voyageai.Client()

DATASET_REVISION = "3a9e8d226c127392ce81dca29d09f33f0ce3247d"

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def partial_arg_parser():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--output-dir",
        type=str,
        help="Directory in which to place JSON files with completions. The default is root_dataset-lang-model_name-temperature-reworded",
    )

    args.add_argument(
        "--output-dir-prefix", type=str, help="Prefix for the output directory"
    )

    args.add_argument(
        "--use-local",
        action="store_true",
        help="Use this flag when running from local prompts.",
    )

    # Reuired when use local is passed
    args.add_argument(
        "--dataset",
        type=str,
        required="--use-local" in sys.argv,
        help="The local dataset in JSON format to get from this computer.",
    )
    # Only required when use local is not passed
    args.add_argument(
        "--lang",
        type=str,
        required="--use-local" not in sys.argv,
        help="Target language for completions",
    )
    args.add_argument(
        "--root-dataset",
        type=str,
        required="--use-local" not in sys.argv,
        help="either mbpp or humaneval",
    )
    args.add_argument("--temperature", type=float, required=True)
    args.add_argument(
        "--input-start-index",
        type=int,
        help="Index into the dataset. If omitted, starts from the beginning",
    )
    args.add_argument(
        "--input-limit", type=int, help="Number of items to process from the dataset"
    )
    args.add_argument("--completion-limit", type=int, default=200)
    args.add_argument(
        "--batch-size", type=int, default=16, help="Number of completions to batch"
    )
    args.add_argument(
        "--prompt-num", type=int, help="number of examples"
    )
    args.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    args.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value for sampling",
    )
    return args


def make_main(args, model_name, gen_completions):

    assert "-" not in model_name, "Model name must not have hyphens"

    if args.output_dir is None:
        args.output_dir = (
            (
                f"{args.root_dataset}-{args.lang}-{model_name}-{args.temperature}-reworded"
            )
            if not args.use_local
            else (
                f"{args.dataset.split('/')[-1].split('.')[0]}-{model_name}-{args.temperature}-reworded"
            )
        )

    if args.output_dir_prefix is not None:
        args.output_dir = f"{args.output_dir_prefix}/{args.output_dir}"

    exp_dir = Path(args.output_dir)
    if not exp_dir.exists():
        exp_dir.mkdir()

    if args.use_local:
        problems = datasets.load_dataset(
            "json", data_files=args.dataset, split="train")
    else:
        problems = datasets.load_dataset(
            "nuprl/MultiPL-E", f"{args.root_dataset}-{args.lang}", revision=DATASET_REVISION, split="test"
        )

    start_index = args.input_start_index if args.input_start_index is not None else 0
    stop_index = min(
        len(problems),
        start_index + args.input_limit
        if args.input_limit is not None
        else len(problems),
    )
    start_index = args.input_start_index if args.input_start_index is not None else 0
    stop_index = min(
        len(problems),
        start_index + args.input_limit
        if args.input_limit is not None
        else len(problems),
    )
    problems = problems.select(range(start_index, stop_index))

    # Read all existing completions
    all_completions = dict(read_completions(
        exp_dir, args.temperature, args.top_p, args.max_tokens, problem) for problem in problems)

    # Generate a list of prompts, including multiple copies when needed.
    problem_list = []
    stop: List[str] = None
    if args.prompt_prefix is not None:
        train_examples = read_examples("/project/def-fard/zixiao/dataset/Rcombine/train.jsonl")
        #corpus = [ex.target for ex in train_examples]
        #bm25 = BM25Okapi(corpus)
        # 将 args.prompt_prefix 转换为 int 作为返回的 top n 数量
        top_n = int(args.prompt_prefix)
    for completions in all_completions.values():

        if stop is None:
            stop = completions["stop_tokens"]
        else:
            assert stop == completions["stop_tokens"], "Stop tokens must be the same for all completions"

        assert completions["temperature"] == args.temperature, "Temperature must be the same for all completions"

        if len(completions["completions"]) >= args.completion_limit:
            continue

        num_new_completions = args.completion_limit - \
            len(completions["completions"])

        if args.prompt_prefix is not None:
            print("start find prompt")
            # completions["prompt"] 作为查询问题，检索 top_n 个最相似的 few-shot 示例
            few_shot_results = get_top_n_similar_embedding_from_json(completions["prompt"], train_examples, top_n=top_n)
            print("got prompt")
            # 对每个候选例子将 docstring 和 code 拼接，并用换行符分隔
            few_shot_examples = "\n\n".join([combine_example(ex) for ex, score in few_shot_results])
            # 将 few-shot 示例作为前缀，加上当前的查询问题作为完整 prompt
            prompt = "Example R function:\n\n" + few_shot_examples + "\n\nend of demo\n\n" + completions["prompt"]
            print(prompt)
        else:
            prompt = completions["prompt"]
        item = {"prompt": prompt, "name": completions["name"]}

        problem_list.extend([item for _ in range(num_new_completions)])

    # Break problem_list into batches of size args.batch_size.
    problem_list = [problem_list[i:i+args.batch_size]
                    for i in range(0, len(problem_list), args.batch_size)]

    for batch in tqdm(problem_list, unit="batch"):
        new_completions = gen_completions(
            prompts=[item["prompt"] for item in batch],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=stop
        )
        modified_problems = set()
        for item, a_completion in zip(batch, new_completions):
            # if a_completion is just a string, run normal completion
            if isinstance(a_completion, str):
                completion = a_completion
            else:
                # assert it's a 3-tuple
                assert len(
                    a_completion) == 3, "Completion must be a 3-tuple or a string"
                completion, logprob, num_tokens = a_completion
                if "tokens_info" not in all_completions[item["name"]]:
                    all_completions[item["name"]]["tokens_info"] = []
                all_completions[item["name"]]["tokens_info"].append(
                    {"cumulative_logprob": logprob, "len": num_tokens})

            all_completions[item["name"]
                            ]["completions"].append(completion)
            modified_problems.add(item["name"])

        for name in modified_problems:
            with gzip.open(exp_dir / f"{name}.json.gz", "wt") as f:
                f.write(json.dumps(all_completions[name]))


def read_completions(exp_dir, temperature, top_p, max_tokens, problem):
    problem_filename = exp_dir / f"{problem['name']}.json.gz"
    if problem_filename.exists():
        with gzip.open(problem_filename, "rt") as f:
            existing = json.loads(f.read())
            return (existing["name"], existing)

    new_completions = {
        "name": problem["name"],
        "language": problem["language"],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "prompt": problem["prompt"],
        "tests": problem["tests"],
        "completions": [],
        "stop_tokens": problem["stop_tokens"],
    }
    return (new_completions["name"], new_completions)


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=js['code_tokens']
            nl=js['docstring_tokens']           
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        ) 
            )
    return examples

def combine_example(example):
    """
    将一个候选 Example 的 docstring 和 code 合并为一个字符串，
    作为 few shot learning 中的示例使用。
    """
    nl = example.target
    code = example.source
    code=' '.join(code).replace('\n',' ')
    code=' '.join(code.strip().split())
    nl=' '.join(nl).replace('\n','')
    nl=' '.join(nl.strip().split())   
    return f"Docstring: {nl}\nCode: {code}"

def get_top_n_similar(query, examples, bm25, top_n=1):
    """
    对一个查询 query，从 Example 对象列表中返回 top_n 个最相似的候选项。
    返回结果为列表，每个元素为 (Example 对象, 得分) 的元组。
    """
    scores = bm25.get_scores(query)
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:top_n]
    return [(examples[i], scores[i]) for i in top_indices]

def get_top_n_similar_embedding(query, examples, top_n):
    """
    对一个查询 query，从 examples 列表中返回 top_n 个最相似的候选项，
    基于 voyageai 的 "voyage-code-3" 模型计算文本的 embedding，
    并通过余弦相似度进行相似性比较。
    返回结果为列表，每个元素为 (候选项, 相似度得分) 的元组。
    前提是已导入 voyageai 库并创建客户端：
        import voyageai
        voyageai.api_key = "api-key"
        vo = voyageai.Client()
    """
    time.sleep(0.03)
    # 获取查询的 embedding
    query_embedding = vo.embed([query], model="voyage-code-3").embeddings[0]
    example_texts = [clean_tokens(ex.target) for ex in examples]
    # 获取 examples 的 embeddings，假设 examples 为文本列表
    example_embeddings = []
    batch_size = 128
    # 批量调用 API 处理 examples
    for i in range(0, len(example_texts), batch_size):
        batch = example_texts[i:i + batch_size]
        batch_embeddings = vo.embed(batch, model="voyage-code-3").embeddings
        example_embeddings.extend(batch_embeddings)
        # 每个批次后暂停，控制请求速率
        time.sleep(0.03)

    # 计算每个 example 与 query 的余弦相似度
    query_vec = np.array(query_embedding)
    scores = []
    for emb in example_embeddings:
        emb_vec = np.array(emb)
        cosine_score = np.dot(query_vec, emb_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec))
        scores.append(cosine_score)

    scores = np.array(scores)
    # 按相似度从高到低排序
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:top_n]

    # 返回 top_n 个结果，形式为 (example, score) 的元组
    return [(examples[i], scores[i]) for i in top_indices]

def get_top_n_similar_embedding_from_json(query, examples, top_n):
    """
    从指定的 JSON 文件中读取预计算的 example_embeddings，
    对查询 query 计算 embedding，然后与 example_embeddings 计算余弦相似度，
    返回 top_n 个最相似的候选项，形式为 (Example, score) 的元组。
    
    参数：
      query: 查询文本（字符串）。
      examples: 原始 Example 对象列表（顺序需与 JSON 中保存的 embeddings 一致）。
      json_filename: 包含预计算 embeddings 的 JSON 文件路径。
      top_n: 返回最相似候选项的数量。
      
    返回：
      返回一个列表，每个元素为 (Example, cosine_score) 的元组。
      
    前提：
      - 已初始化 voyageai 客户端 vo。
    """
    # 从 JSON 文件中读取 embeddings
    with open("/project/def-fard/zixiao/MultiPL-E/multipl_e/embeddings.json", "r", encoding="utf-8") as f:
        example_embeddings = json.load(f)
    print("loaded")    
    # 计算查询文本的 embedding
    
    with open("/project/def-fard/zixiao/MultiPL-E/multipl_e/humaneval_R_embeddings.json", "r", encoding="utf-8") as f:
        query_results = json.load(f)
    print("Loaded query embeddings from humaneval_R_embeddings.json.")
    
    # 查找与输入 query 匹配的 query embedding
    query_embedding = None
    for entry in query_results:
        if entry.get("prompt", "").strip() == query.strip():
            query_embedding = entry.get("embedding")
            break
    if query_embedding is None:
        raise ValueError("Query prompt not found in humaneval_R_embeddings.json.")


    query_vec = np.array(query_embedding)
    
    # 计算余弦相似度
    scores = []
    for emb in example_embeddings:
        emb_vec = np.array(emb)
        cosine_score = np.dot(query_vec, emb_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec))
        scores.append(cosine_score)
        
    scores = np.array(scores)
    # 按相似度从高到低排序
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:top_n]
    
    return [(examples[i], scores[i]) for i in top_indices]
