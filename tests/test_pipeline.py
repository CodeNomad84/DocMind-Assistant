# Pipeline کامل: PDF → Answer

from core.loader import PDFLoader
from core.chunker import TextChunker
from core.embeddings import EmbeddingGenerator
from core.vector_store import FAISSVectorStore
from core.retriever import Retriever
from core.llm import OpenAIClient
from core.qa_chain import QAChain

# ۱. Indexing (یکبار اجرا می‌شه)
loader = PDFLoader("data/raw/contract.pdf")
pages = loader.load()

chunker = TextChunker()
chunks = chunker.chunk_pages(pages)

generator = EmbeddingGenerator()
embeddings = generator.embed_chunks(chunks)

store = FAISSVectorStore(embedding_dim=768)
store.add_embeddings(chunks, embeddings)
store.save("data/processed/vector_store")

# ۲. Query (هر بار که کاربر سوال می‌پرسه)
retriever = Retriever(generator, store)
llm = OpenAIClient()
chain = QAChain(retriever, llm)

result = chain.ask("قرارداد چند ساله است؟")
print(result["answer"])
