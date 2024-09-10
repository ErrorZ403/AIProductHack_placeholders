import logging
from typing import Any, Dict, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document 
from config import configs
import os


# починить откуда прокидываю init
class FaissDBManager:
    def __init__(self, name_of_model: str, index_file: Optional[str]) -> None:
        logging.info('Initialization')
        self.name_of_model = name_of_model
        self.index_file = index_file
        self.embedding_function: Optional[Any] = None
        self.vector_store: Optional[FAISS] = None

    async def setup(self) -> None:
        await self.setup_embedding_function()
        await self.setup_faiss_index()

    async def setup_embedding_function(self) -> None:
        logging.info('setup_embedding_function')
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.name_of_model)

    async def setup_faiss_index(self) -> None:
        logging.info('setup_faiss_index')
        if self.index_file:
            self.vector_store = FAISS.load_local(self.index_file, self.embedding_function, allow_dangerous_deserialization=True)
            logging.info(f'Loaded FAISS index from {self.index_file}')
        else:
            self.vector_store = FAISS(embedding_function=self.embedding_function, index=None)
            logging.info('Initialized new FAISS index')

    async def add_documents_to_index(self, documents: List[Dict[str, Any]]) -> None:
        logging.info('---Adding documents to FAISS index---')
        docs = [Document(page_content=doc['content'], metadata={'date': doc['date'], 'project_uid': doc['project_uid']})
                for doc in documents]

        # Добавляем документы в FAISS индекс
        self.vector_store.add_documents(docs)
        logging.info('Documents added to FAISS index.')

    async def query(self, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        logging.info('---Querying FAISS index---')
        results = self.vector_store.similarity_search(query_text, k=n_results)

        return [{"content": result.page_content, "metadata": result.metadata} for result in results]

    async def save_index(self, file_path: str) -> None:
        logging.info(f'Saving FAISS index to {file_path}')
        self.vector_store.save_local(file_path)

    async def delete_document_by_id(self, document_id: int) -> None:
        logging.warning("FAISS (through Langchain) does not natively support deleting specific entries.")

    async def update_document(self, document_id: int, new_content: Any, project_uid: str, content_date: str) -> None:
        logging.warning("Updating specific entries requires re-indexing in FAISS (through Langchain).")

    async def get_collection_length(self) -> int:
        return len(self.vector_store.index_to_docstore_id)


async def get_faiss_manager(model_name: Optional[str] = None, index_file: Optional[str] = None) -> FaissDBManager:
    try:
        logging.info('Connecting to FAISS through Langchain...')

        faiss_manager = FaissDBManager(name_of_model=configs.db_config.name_of_model, index_file="data_chanks/e5large_256_64_faiss" if os.path.exists("data_chanks/e5large_256_64_faiss") else None)
        await faiss_manager.setup()

        return faiss_manager
    except Exception as e:
        logging.exception(f'Failed to connect to FAISS: {e}')
        return None
