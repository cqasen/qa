import time

import streamlit as st

import utils
from config import Config


def qa_page(embeddings, db):
    # Streamlit 应用
    st.title("💬 知识问答")
    disabled = False
    with st.sidebar:
        options = []
        if Config.dashscope_api_key is not None:
            other = ['qwen/qwen-max']
            options.extend(other)
        options.extend(Config.pipeline_model_names)

        pipeline_model_name = st.selectbox("模型选择", options=options)

    if pipeline_model_name is None:
        disabled = True
        st.warning("请先选择问答模型")

    # 用户输入问题
    question_input = st.text_input(label="请输入您的问题:", placeholder="请输入您的问题")

    # 当用户点击按钮时执行
    if st.button("提问", disabled=disabled):
        if question_input == "":
            warning_tips = st.warning("请输入您的提问")
            time.sleep(2)
            warning_tips.empty()
            return
        info_tips = st.info("正在为您回答问题...")
        page_content = utils.load_similarity_search(db, question_input)
        if page_content == "":
            time.sleep(1)
            info_tips.empty()
            warning_tips = st.warning("对不起，我不知道这个问题的答案。请先上传相关的知识库文档文件")
            time.sleep(2)
            warning_tips.empty()
            return

        # print(question_input)
        # print(page_content)
        # 调用缓存的 QA 模型
        # 加载 QA 模型
        if pipeline_model_name in Config.pipeline_model_names:
            qa_pipeline = utils.load_qa_pipeline(pipeline_model_name)
            result = qa_pipeline(question=question_input, context=page_content, max_answer_len=100, max_seq_len=512)
        else:
            response = utils.qwen_chat(query=question_input, content=page_content)
            result = {
                "answer": response
            }

        info_tips.empty()
        info_tips.info("以下是给出的答案")
        # 显示结果
        answer = result['answer']
        if answer == "。":
            st.warning("对不起，我不知道这个问题的答案。")
        else:
            st.write(f"问题: {question_input}")
            st.write(f"答案: {answer}")
