import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import torch

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"],
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset, llm, embeddings):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness,
        answer_similarity
    ],
    llm=llm,
    embeddings=embeddings
  )
  return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="Path to validation dataset")
    parser.add_argument("--model_name", type=str, help="Name of model from huggingface")
    parser.add_argument("--model_name_llm", type=str, help="LLM Model name from hf")
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'batch_size': args.batch_size}

    embeddings = HuggingFaceEmbeddings(model_name=args.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    ragas_dataset = pd.read_csv(args.dataset_path)
    ragas_dataset = ragas_dataset.rename(columns={"ground_truths": "ground_truth"})
    ragas_dataset["contexts"]=ragas_dataset["contexts"].apply(lambda x : [x])
    ragas_dataset = Dataset.from_pandas(ragas_dataset)
    
    llm = VLLM(
        model='Qwen/Qwen2-7B',
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    result = evaluate_ragas_dataset(ragas_dataset, llm, embeddings)
    print(result)

if __name__ == "__main__":
    main()