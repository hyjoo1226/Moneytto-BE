import bs4
import os, json
import urllib.request

from konlpy.tag import Okt
from langchain_community.document_loaders import WebBaseLoader

def load_indexing_news(news_list):
    docs = []  # 성공적으로 로드된 문서를 저장
    for url in news_list:
        try:
            # 웹 페이지 로더 설정
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        "div",
                        attrs={
                            "class": [
                                "newsct_article _article_body",
                                "media_end_head_title",
                                "view_con",
                                "list_contents_container",
                                "article_container_layout",
                                "user-container",
                                "section_left_container",
                            ]
                        },
                    )
                ),
            )
            docs.extend(loader.load())
        except Exception as e:
            print(f"Invalid news format")
    return docs


def get_naver_news_with_kewords(search_keywords, num=20):
    CLIENT_ID = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    
    encText = urllib.parse.quote(search_keywords)
    url = "https://openapi.naver.com/v1/search/news?query=" + encText + f"&display={num}" # JSON 결과
    
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        # print(response_body.decode('utf-8'))
        response_json = json.loads(response_body.decode('utf-8'))
        news_list = []
        for item in response_json['items']:
            news_list.append(item['link'])
        return news_list
    else:
        print("Error Code:" + rescode)
        return 'error'



def generate_search_keywords_with_konlpy(question):           
    stopwords = ["최근", "구애", "고려", "경제", "트렌드", "전망", "관계", "국가",
                 "관련", "타이밍", "설명", "분석", "매수", "매도", "인하", "상승",
                 "예측", "종목", "최대", "최소"]
    
    from kiwipiepy import Kiwi

    # Kiwi 객체 생성
    kiwi = Kiwi()
    result = kiwi.analyze(question)
    
    nouns = []
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and (pos.startswith('NN') or pos.startswith('SL')):
            nouns.append(token)
    
    for stopword in stopwords:
        if stopword in nouns:
            nouns.remove(stopword)
    # 중복 제거 및 검색어 조합
    keywords = set(nouns)  # 중복 제거아 그렇구나. 주식에도 투자할까 하는데 최근 미국과 중국간의 관계는 어떤지 분석해서 최대 수혜를 받을 종목을 알려줄 수 있어?
    search_query = ' '.join(keywords)
    print("생성된 검색어:", search_query)
    
    return search_query
    
def get_naver_news_list(question, choice):
    economic_news_list = []
    keywords = ["정치 경제 ", "금리  ", "해외 투자 ", "대출 ", "주요 지수 ", ""]
    keywords += generate_search_keywords_with_konlpy(question)
    for keyword in keywords:
        try:
            economic_news_list += get_naver_news_with_kewords(keyword + choice, num=4)
        except:
            print(f'Error occurred while fetching news for keyword: {keyword}')
    economic_news_list = set(economic_news_list)
    print('num of news: ', len(economic_news_list))
    return economic_news_list
    
def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)



