import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Now this import should work
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

from get_docs import format_docs, load_indexing_news, get_naver_news_with_kewords


# load_dotenv()
# open_ai_key = os.environ.get("open_ai_key")

# # upstage models
# embedding_upstage = UpstageEmbeddings(model="embedding-query")

# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# pc = Pinecone(api_key=pinecone_api_key)

def get_prompt_for_type(customer_type):
    first_template = f"""
    당신은 금융 상담 전문가이며, 고객의 투자성향에 맞는 투자상품을 참조문서에 기반해 추천하는 역할을 맡고 있습니다.


    고객의 투자성향은 {customer_type}형입니다.
    """ + """
    1. 참조문서에서 고객의 투자성향을 찾아 추천되는 투자상품을 찾습니다.
    2. 추천하는 투자상품 종류를 나열하고, 이후 투자상품과 그 특징을 설명합니다.
        - 어려운 단어는 사용자가 이해하기 쉽도록 풀어서 설명하고 쉬운 용어로 대체하세요.
    3. 모르면 모른다고 하세요. 모든 답변은 명확한 근거를 가진 사실에 기반합니다. 예) 질문을 이해하기 어렵습니다. 잘 모르겠습니다.
    4. 참조문서와 참조뉴스 기반으로 사용자의 질문에 틀린 내용이 있다면 수정해서 답변하세요.
    5. 사용자의 개인정보 및 민감정보(주민등록번호, 카드정보 등)가 질문에 들어오면 '개인의 민감한 정보를 입력하지 마세요!'라는 경고를 띄우고 답변 생성하는 것을 멈춥니다.
    6. 답변 형식에 맞추세요.

    
    질문: {question}
    참조문서: {context_pdf}
    답변 형식:
    #### 💚 머니또님의 투자성향은 [...]입니다.
    추천하는 투자상품은 ... 입니다.
    1. 투자상품:
        - 특징 및 리스크:
    2. 투자상품:
        - 특징 및 리스크:
    ...

    """
    # 이전 메시지: {chat_history}

    # 새로운 PromptTemplate 생성
    prompt_template_after_survey = PromptTemplate(
        input_variables=["context_pdf", "question"],#, "chat_history"],  # 변수 그대로 유지
        template=first_template  # 새 템플릿 적용
    )

    # HumanMessagePromptTemplate 생성
    updated_human_prompt1 = HumanMessagePromptTemplate(prompt=prompt_template_after_survey)

    # ChatPromptTemplate 객체 생성
    updated_prompt1 = ChatPromptTemplate(messages=[updated_human_prompt1])
    
    return updated_prompt1

def get_prompt_for_recommend(choice):
    recommend_template = f"""
    당신은 금융 상담 전문가이며, 고객의 질문에 맞는 대답을 하고 이에 대한 근거를 참조문서에 기반해 대답하는 역할을 맡고 있습니다.

    고객이 원하는 금융투자상품은 {choice}입니다.
    """ + """
    1. 참조문서를 참고해서 사용자가 원하는 금융투자상품과 경제 지수 및 요인과의 관계를 파악하세요.
        - 금융투자상품이 큰 수익성을 가지는 시점과 리스크를 가지는 시점을 분석하세요.
    2. 1번 결과와 참조뉴스를 참고하여 사용자의 질문에 부합하는 답변을 하세요.
        - 상품/종목 선택 기준은 명확하고 논리적이어야 하며, 장단점 및 리스크가 균형 있게 포함되도록 작성하세요.
        - 추천 이유는 뉴스를 통해 현재 경기 상황을 파악하고 요약한 결과를 보고 최신 동향을 반영합니다. 이후 이득을 취할 것으로 예상되는 종목을 추천하고, 그 근거를 참조뉴스에서 찾아 구체적으로 덧붙이세요.
        - 만약, 사용자가 선택한 금융투자상품이 리스크를 가지는 시점이라면 안정적인 다른 투자상품(예적금, 채권, 펀드 등등)을 추천합니다.
    3. 모르면 모른다고 하세요. 모든 답변은 명확한 근거를 가진 사실에 기반합니다. 예) 질문을 이해하기 어렵습니다. 잘 모르겠습니다.
    4. 참조문서와 참조뉴스 기반으로 사용자의 질문에 틀린 내용이 있다면 수정해서 답변하세요.
    5. 사용자의 개인정보 및 민감정보(주민등록번호, 카드정보 등)가 질문에 들어오면 '개인의 민감한 정보를 입력하지 마세요!'라는 경고를 띄우고 답변 생성하는 것을 멈춥니다.
    6. 답변 형식에 맞추세요.


    
    질문: {question}
    참조문서: {context_pdf}
    참조뉴스: {context_news}
    답변 형식:
    - 현재 경제 상황:
    - 선택하신 금융투자상품:
    - 추천 종목:
    - 종목 추천 이유:

    """
    # 이전 메시지: {chat_history}
    # 새로운 PromptTemplate 생성
    updated_prompt_template = PromptTemplate(
        input_variables=["context_pdf", "context_news", "question"],#, "chat_history"],  # 변수 그대로 유지
        template=recommend_template  # 새 템플릿 적용
    )

    # HumanMessagePromptTemplate 생성
    updated_human_prompt2 = HumanMessagePromptTemplate(prompt=updated_prompt_template)

    # ChatPromptTemplate 객체 생성
    updated_prompt2 = ChatPromptTemplate(messages=[updated_human_prompt2])
    return updated_prompt2


def set_rag_chain_for_type(customer_type, open_ai_key, pc):#, chat_history):
    
    # upstage models
    embedding_upstage = UpstageEmbeddings(model="embedding-query")
    
    pdf_vectorstore = PineconeVectorStore(
        index=pc.Index("index-pdf"), embedding=embedding_upstage
    )
    
    pdf_retriever = pdf_vectorstore.as_retriever(
        search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
        search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
    )
    
    # ReRanker: CrossEncoder
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)

    pdf_retriever_compression = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=pdf_retriever
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0,
                    openai_api_key=open_ai_key)

    # 체인을 생성합니다.
    rag_chain = (
        {"context_pdf": pdf_retriever_compression, "question": RunnablePassthrough()}#, "chat_history": chat_history} #
        | get_prompt_for_type(customer_type)
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke("나의 투자성향에 맞는 투자상품을 알려줘.")



def set_rag_chain_for_recommend(question, choice, open_ai_key, pc):#, chat_history):
    # upstage models
    embedding_upstage = UpstageEmbeddings(model="embedding-query")
    
    pdf_vectorstore = PineconeVectorStore(
        index=pc.Index("index-pdf"), embedding=embedding_upstage
    )
    
    news_vectorstore = PineconeVectorStore(
        index=pc.Index("index-news"), embedding=embedding_upstage
    )
    
    pdf_retriever = pdf_vectorstore.as_retriever(
        search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
        search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
    )

    # 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
    news_retriever = news_vectorstore.as_retriever(
        search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
        search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
    )

    # ReRanker: CrossEncoder
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)

    pdf_retriever_compression = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=pdf_retriever
    )
    news_retriever_compression = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=news_retriever
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0,
                    openai_api_key=open_ai_key)

    # 체인을 생성합니다.
    rag_chain = (
        {"context_news": news_retriever_compression | format_docs, "context_pdf": pdf_retriever_compression, "question": RunnablePassthrough()}#, "chat_history": chat_history}
        | get_prompt_for_recommend(choice)
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)