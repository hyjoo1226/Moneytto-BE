import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

from module.get_docs import get_naver_news_list, load_indexing_news

load_dotenv()

# upstage models
embedding_upstage = UpstageEmbeddings(model="embedding-query")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "index-news"
# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start>> news vectorstore")

choice = ''
news_list = get_naver_news_list(choice)
news_docs = load_indexing_news(news_list)
    
text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

# Embed the splits
splits = text_splitter.split_documents(news_docs)
print("Splits:", len(splits))

# 벡터화
vectorstore = PineconeVectorStore.from_documents(documents=splits, embedding=embedding_upstage,# embedding=OpenAIEmbeddings(api_key=open_ai_key),
                                                index_name=index_name)

print("end>> news vectorstore")