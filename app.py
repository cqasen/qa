import streamlit as st
from streamlit_option_menu import option_menu

import utils
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from webui_pages.qa.qa import qa_page

if __name__ == '__main__':
    st.set_page_config(
        page_title="本地知识库问答",
        page_icon="🤔",
        layout="wide",
        initial_sidebar_state="auto",
    )
    # 初始化模型
    utils.init_model()

    pages = {
        "问答": {
            "icon": "chat",
            "func": qa_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        }
    }
    with st.sidebar:
        st.image("./images/logo.png", width=278)
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]
        default_index = 0
        selected_page = option_menu(
            "功能导航",
            options=options,
            icons=icons,
            default_index=default_index,
        )
    if selected_page in pages:
        pages[selected_page]["func"]()
