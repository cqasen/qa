import time

import pandas as pd
import streamlit as st

import utils


def knowledge_base_page():
    # 通过侧边栏上传 txt 文件
    st.subheader("📝知识库配置")

    embeddings = utils.load_embeddings()
    db = utils.load_chroma_db(embeddings)

    with st.sidebar:
        with st.expander("清空知识库配置", False):
            st.title("📝知识库配置")
            if st.button("清空知识库"):
                result = db.get()
                ids = result.get("ids")
                if ids:
                    db.delete(ids=ids)
                succ_tips = st.success("清空知识库")
                time.sleep(1)
                succ_tips.empty()

    with st.expander("上传知识库文件", False):
        uploaded_file_type = [
            "txt",
            "pdf",
            # "jpg",
            # "png"
        ]
        uploaded_file = st.file_uploader("上传相关的知识库文档文件", type=uploaded_file_type)
        if st.button("上传文件"):
            if uploaded_file is None:
                sidebar_warning_tips = st.warning("请选择要上传的文件")
                time.sleep(2)
                sidebar_warning_tips.empty()

            else:
                # 读取文件内容

                sidebar_tips = st.info(f"您上传的文件名为 {uploaded_file.name},\n文件正在处理中...")
                utils.load_uploader_file_vector_db(uploaded_file, embeddings)

                sidebar_tips.empty()
                sidebar_tips.info("文档加载完成")

    result = db.get()
    ids = result.get("ids")
    df = pd.DataFrame(result)
    del_tips = None
    if ids:
        ids = st.text_input("请输入要删除的ids", "")
        id_list = [id.strip() for id in ids.split(',') if id.strip() != '']
        if st.button("删除"):
            del_tips = st.empty()
            if len(id_list) > 0:
                db.delete(ids=id_list)
                df.drop(df[df['ids'].isin(id_list)].index, inplace=True)
                del_tips.success("删除成功")
            else:
                del_tips.warning("请选择要删除的ids")

    result_df = df.filter(items=['ids', 'metadatas', 'documents'])
    st.dataframe(result_df, hide_index=True)
    if del_tips:
        time.sleep(1)
        del_tips.empty()
