import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'guide-to-meditation'

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension= 768,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

embeddings= OllamaEmbeddings(model="nomic-embed-text")

pdf_path = "00_Mullin_Appendix_10.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap = 200,
    length_function=len
)


chunks= textSplitter.split_documents(pages)

if chunks:
    print("\n--- First Chunk Preview ---")
    print(chunks[0].page_content[:300] + "...")
    print(f"Metadata: {chunks[0].metadata}")
    print("---------------------------\n")



vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY
)

print("All chunks successfully embedded and stored in Pinecone!")