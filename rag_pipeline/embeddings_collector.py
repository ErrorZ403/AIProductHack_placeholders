from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS

def load_text(path, chunk_size=256, chunk_overlap=64):
    loader = TextLoader("hmao_npa.txt")
    base_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(base_docs)

    return docs

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--doc_path", type=str, help="Path to documents")
    parser.add_argument("--model_name", type=str, help="Name of model from huggingface")
    parser.add_argument("--save_path", type=str, help="Path for saving embeddings", default="vectorbd")
    parser.add_argument("--chunk_size", type=int, help="Chunk size for document splittings", default=256)
    parser.add_argument("--chunk_overlap", type=int, help="Overlap between chunks", default=64)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()
    
    docs = load_text(args.doc_path, args.chunk_size, args.chunk_overlap)
    
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'batch_size': args.batch_size}

    vectorstore = FAISS.from_documents(docs, 
                                    HuggingFaceEmbeddings(model_name=args.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs))
    vectorstore.save_local(args.save_path)

if __name__ == "__main__":
    main()

    
