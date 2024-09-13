import argparse
import faiss
from operator import itemgetter
from ragas import evaluate
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import torch

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import VLLM
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import RetrievalQA


def load_text(path, chunk_size=256, chunk_overlap=64):
    loader = TextLoader(path)
    base_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(base_docs)

    return docs

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--doc_path", type=str, help="Path to documents")
    parser.add_argument("--dataset_path", type=str, help="Path to validation dataset")
    parser.add_argument("--model_name", type=str, help="Name of model from huggingface")
    parser.add_argument("--emb_path", type=str, help="Path to embeddings")
    parser.add_argument("--model_name_llm", type=str, help="LLM Model name from hf")
    parser.add_argument("--path_to_save", type=str, help="Path for saving rag results", default="rag_results.csv")
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--bm25", type=bool, default=False)
    parser.add_argument("--rerank", type=bool, default=False)

    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    model_kwargs = {'device': device, "trust_remote_code": True}
    encode_kwargs = {'batch_size': args.batch_size}
    
    docs = load_text(args.doc_path, 256, 0)

    embeddings = HuggingFaceEmbeddings(model_name=args.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    vectorstore = FAISS.load_local(args.emb_path, embeddings, allow_dangerous_deserialization=True)
    base_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k" : 10})
    
    if args.bm25:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        base_retriever = EnsembleRetriever(retrievers=[bm25_retriever, base_retriever], weights=[0.75, 0.25])

    if args.rerank:
        model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
        
        model = HuggingFaceCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual", model_kwargs=model_kwargs)
        compressor = CrossEncoderReranker(model=model, top_n=3)
        base_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    
    
    template = """Ответьте на вопрос, опираясь только на следующий контекст. Если вы не можете ответить на вопрос, опираясь на контекст, пожалуйста, ответьте «Я не знаю»:

    ### КОНТЕКСТ
    {context}

    ### ВОПРОС
    ВОПРОС: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = VLLM(
        model=args.model_name_llm,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=10000,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )

    eval_dataset = pd.read_excel(args.dataset_path)[:50]
    eval_dataset = Dataset.from_pandas(eval_dataset)
    ragas_dataset = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
    qa_ragas_dataset.to_csv(args.path_to_save)
    print('DONE')

if __name__ == "__main__":
    main()