import bs4
import os, json
import urllib.request

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
            print(f"Failed to load content from {url}: {e}")
    return docs


def get_naver_news_with_kewords(search_keywords, num=20):
    
    CLIENT_ID = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    
    encText = urllib.parse.quote(search_keywords)
    url = "https://openapi.naver.com/v1/search/news?query=" + encText + f"&display={num}" # JSON 결과
    # url = "https://openapi.naver.com/v1/search/news.xml?query=" + encText # XML 결과
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
        print("검색된 뉴스: ", news_list)
        return news_list
    else:
        print("Error Code:" + rescode)
        return 'error'
    
    
def get_naver_news_list(choice):
    economic_news_list = []
    keywords = ["정치 경제 ", "금리 전망 ", "미국 경제 ", "해외 투자 ", "대출 ", "환율 ", "주요 지수 ", ""]
    for keyword in keywords:
        economic_news_list += get_naver_news_with_kewords(keyword + choice, num=5)
    economic_news_list = set(economic_news_list)
    print('num of news: ', len(economic_news_list))
    return economic_news_list
    
def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)