import datetime
import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
from langchain.agents import tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi

import utils
from config import Config


def qa_page():
    # Streamlit 应用
    st.title("💬 知识问答")

    os.environ["DASHSCOPE_API_KEY"] = Config.dashscope_api_key
    os.environ["AMAP_TOKEN"] = Config.amap_token
    llm = Tongyi(model_name="qwen-max")

    ORDER_1 = "20230611ABC"
    ORDER_2 = "20230611EFG"

    ORDER_1_DETAIL = {
        "order_number": ORDER_1,
        "status": "已发货",
        "shipping_date": "2023-01-03",
        "estimated_delivered_date": "2023-01-05",
    }

    ORDER_2_DETAIL = {
        "order_number": ORDER_2,
        "status": "未发货",
        "shipping_date": None,
        "estimated_delivered_date": None,
    }

    @tool("Search Order")
    def search_order(input: str) -> str:
        """一个帮助用户查询最新订单状态的工具，并且能处理以下情况：
        1. 在用户没有输入订单号的时候，会询问用户订单号
        2. 在用户输入的订单号查询不到的时候，会让用户二次确认订单号是否正确
        """
        pattern = r"\d+[A-Z]+"
        match = re.search(pattern, input)

        order_number = input
        if match:
            order_number = match.group(0)
        else:
            return "请问您的订单号是多少？"
        if order_number == ORDER_1:
            return json.dumps(ORDER_1_DETAIL)
        elif order_number == ORDER_2:
            return json.dumps(ORDER_2_DETAIL)
        else:
            return f"对不起，根据{input}没有找到您的订单"

    @tool("get_baike", return_direct=True)
    def get_baike(query: str) -> str:
        """你是Google小助手，可以查询任何人物，事件，物品的百科信息。
        """
        # 定义从谷歌网页中搜索出想要信息的模版
        template = """在 >>> 和 <<< 直接是来自Google的原始搜索结果.
                   请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
                   请使用如下格式：
                   Extracted:<answer or "找不到">
                   >>> {requests_result} <<<
                   Extracted:
                   """
        PROMPT = PromptTemplate(
            input_variables=["query", "requests_result"],
            template=template,
        )
        # 将把PROMPT传给LLMRequestsChain
        request_chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))
        # 对应的搜索词语，通过query参数传入，对应的原始搜索结果默认放入requests_result，有对应的占位符替换
        question = query
        inputs = {
            "query": question,
            "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
        }
        # 运行一下就会通过OpeenAI提取搜索结果
        result = request_chain(inputs)
        print("\n")
        print(result)
        return result['output']

    @tool("get_product", return_direct=True)
    def get_product(query: str) -> str:
        """从易宠商城中推荐宠物用品，食品，药品
        1.返回信息一行一条，格式对齐，要求有商品标题，价格，是否有货
        """
        # 定义从谷歌网页中搜索出想要信息的模版
        template = """在 >>> 和 <<< 直接是来自易宠商城的原始搜索结果.
                   请把对于问题 '{query}' 的答案从里面提取出来，如果里面没有相关信息的话就说 "找不到"
                   返回数据的格式对齐如下：要求有商品标题，价格，是否有货 空一行,markdown格式
                   请使用如下格式：
                   Extracted:
                   <answer or "找不到">
                   >>> {requests_result} <<<
                   Extracted:
                   """
        PROMPT = PromptTemplate(
            input_variables=["query", "requests_result"],
            template=template,
        )
        # 将把PROMPT传给LLMRequestsChain
        request_chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))
        # 对应的搜索词语，通过query参数传入，对应的原始搜索结果默认放入requests_result，有对应的占位符替换
        question = query
        inputs = {
            "query": question,
            "url": "https://list.epet.com/search/main.html?keyword={0}&version=2.0.0&attrid=".format(
                question.replace(" ", "+"))
        }
        # 运行一下就会通过OpeenAI提取搜索结果
        result = request_chain(inputs)
        print("\n")
        print(result)
        return result['output']

    @tool("get_time")
    def get_time(intput: str) -> str:
        """获取时间,要转化为人眼能够 一眼就识别的格式 Y-m-d H:i:s
        1.如果要获取其他时间，可以先获取今天的时间，再去推算
        2.未来的某个时间也根据今天的时间去推算
        """
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time

    @tool("get_amap_weather")
    def get_amap_weather(intput: str):
        """
        查询城市天气预报
        """
        city_df = pd.read_excel(
            'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/AMap_adcode_citycode.xlsx'
        )
        token = Config.amap_token
        if token is None:
            return "请在环境变量中设置AMAP_TOKEN"

        def get_city_adcode(city_df, city_name):
            filtered_df = city_df[city_df['中文名'].str.contains(city_name, na=False)]
            if len(filtered_df['adcode'].values) == 0:
                raise ValueError(
                    f'location {city_name} not found, availables are {city_df["中文名"]}'
                )
            else:
                return filtered_df['adcode'].values[0]

        adcode = get_city_adcode(city_df, intput)
        url = 'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}'
        full_url = url.format(city=adcode, key=token)

        response = requests.get(full_url)
        data = response.json()

        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            weather = data['lives'][0]['weather']
            temperature = data['lives'][0]['temperature']
            return {'result': f'{intput}的天气是{weather}温度是{temperature}度。'}

    @tool("get_faq")
    def get_faq(query: str):
        """
        优先通过本地知识库问答,小米汽车(su7)答网友100问,劳动法学习实操手册。如果本地知识库回答不了，再通过get_baike进去回答
        """
        embeddings = utils.load_embeddings()
        db = utils.load_chroma_db(embeddings)
        page_content = utils.load_similarity_search(db, query)
        if page_content == "":
            return "对不起，我不知道这个问题的答案。请先上传相关的知识库文档文件"

        response = utils.qwen_chat(query, page_content)
        return response

    # 用户输入问题
    question_input = st.text_input(label="请输入您的问题:", placeholder="请输入您的问题")

    # 当用户点击按钮时执行
    if st.button("提问"):
        if question_input == "":
            warning_tips = st.warning("请输入您的提问")
            time.sleep(2)
            warning_tips.empty()
            return
        info_tips = st.info("正在为您回答问题...")

        tools = [
            get_baike,
            get_product,
            search_order,
            get_amap_weather,
            get_time,
            get_faq
        ]

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        answer = agent.run(question_input)

        # prompt_template = "请用中文回答。提问{问题}"
        # qa = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
        # answer = qa(question_input)

        info_tips.empty()
        info_tips.info("以下是给出的答案")
        # 显示结果
        if answer == "。":
            st.warning("对不起，我不知道这个问题的答案。")
        else:
            st.write(f"问题: {question_input}")
            st.write(f"答案: {answer}")
