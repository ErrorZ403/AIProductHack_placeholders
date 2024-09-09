from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.llms import VLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
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

    parser.add_argument("--doc_path", type=str, help="Path to documents")
    parser.add_argument("--dataset_path", type=str, help="Path to validation dataset")
    parser.add_argument("--model_name", type=str, help="Name of model from huggingface")
    parser.add_argument("--emb_path", type=str, help="Path to embeddings")
    parser.add_argument("--model_name_llm", type=str, help="LLM Model name from hf")
    parser.add_argument("--path_to_save", type=str, help="Path for saving rag results", default="rag_results")
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
    vectorstore = FAISS.load_local(args.emb_path, embeddings, allow_dangerous_deserialization=True)

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
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )

    retrieval_augmented_qa_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": prompt | llm, "context": itemgetter("context")}
    )

    eval_dataset = pd.read_excel(args.dataset_path)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    basic_qa_ragas_dataset = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
    basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset, llm, embeddings)

    basic_qa_ragas_dataset.to_csv(args.path_to_save)
    print(basic_qa_result)

if __name__ == "__main__":
    main()