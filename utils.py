from pathlib import Path

import html2text
import streamlit as st
import torch
from config import Config
from langchain.document_loaders import TextLoader, UnstructuredImageLoader, PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
from transformers import AutoTokenizer
from transformers import pipeline


@st.cache_data()
def init_model():
    # 初始化模型列表
    # 模型名字，这里是两个不同的模型
    # "hfl/chinese-pert-large-mrc",
    # "wptoux/albert-chinese-large-qa",
    # 下载模型
    model_download(Config.pipeline_model_names)


@st.cache_data()
def load_qa_pipeline(model_id):
    from transformers import pipeline
    local_model_id = "./cache_folder/models/{0}".format(model_id.replace("/", "_"))
    # model_id = "./cache_folder/models/hfl_chinese-pert-large-mrc/"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    return pipeline("question-answering", model=local_model_id, tokenizer=model_id, device=device)


@st.cache_data()
def load_embeddings():
    encode_kwargs = {'normalize_embeddings': False}
    model_kwargs = {'device': "cuda" if torch.cuda.is_available() else "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.embedding_model_name,
        cache_folder="./cache_folder/embeddings",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


def save_uploaded_file(uploaded_file, save_path):
    """
    保存上传的文件到指定路径

    参数：
    uploaded_file (file) -- 上传的文件对象
    save_path (str) -- 保存文件的路径

    返回值：
    无

    示例：
    save_uploaded_file(uploaded_file, 'path/to/save/file.txt')
    """
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


def remove_html_tags(input_text):
    """
    去除HTML标签
    :param input_text: 输入的包含HTML标签的文本
    :return: 去除HTML标签后的纯文本
    """
    return html2text.html2text(input_text).replace("\n", "")


def is_image(file_type):
    """
    判断给定文件类型是否为图片类型。

    参数：
    file_type (str)：要判断的文件类型。

    返回值：
    bool：如果给定的文件类型是图片类型，则返回True，否则返回False。
    """
    # 检查MIME类型是否表示图片
    image_mimes = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff", "image/webp"]
    return file_type in image_mimes


def load_uploader_file_vector_db(uploaded_file, embeddings):
    # 保存上传的文件
    uploaded_file_name = f"./runtime/{uploaded_file.name}"
    save_uploaded_file(uploaded_file, uploaded_file_name)

    # 判断文件类型是否为PDF
    is_pdf = uploaded_file.type.endswith("pdf")

    # 判断文件类型是否为图像类型
    is_img = is_image(uploaded_file.type)

    # 根据文件类型选择对应的加载器
    if is_pdf:
        loader = PDFMinerLoader(file_path=uploaded_file_name)
    elif is_img:
        loader = UnstructuredImageLoader(file_path=uploaded_file_name)
    else:
        loader = TextLoader(file_path=uploaded_file_name, autodetect_encoding=True)

    # 加载文档
    documents = loader.load()

    # 清除文档中的HTML标签
    for doc in documents:
        doc.page_content = remove_html_tags(doc.page_content)

    # 创建递归字符文本拆分器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

    # 拆分文档
    splitter_docs = text_splitter.split_documents(documents)

    # 使用拆分后的文档创建Chroma数据库
    vector_store_path = get_vector_store_path()
    db = Chroma.from_documents(documents=splitter_docs, embedding=embeddings, persist_directory=vector_store_path)
    db.persist()
    # 删除上传文件
    Path(uploaded_file_name).unlink()

    # 返回Chroma数据库
    return db


def get_vector_store_path():
    vector_store_path = "./cache_folder/{0}".format(Config.vector_store_path)
    return vector_store_path


def load_similarity_search(db, question_input):
    """
    加载相似度搜索函数

    参数:
        db: 数据库对象
        question_input: 问题输入

    返回:
        无
    """
    # 在此处编写相似度搜索逻辑

    # 使用给定的数据库和问题输入进行相似度搜索
    similar_docs = db.similarity_search(question_input, k=2)
    page_content = ""
    for idx, row in enumerate(similar_docs):
        # 遍历相似文档列表
        # 获取索引和行内容
        # page_content += f"{idx + 1}. {row.page_content}"
        page_content += f"{row.page_content}"
    # 返回相似文档的内容
    return page_content


def model_download(model_name_list):
    # 指定模型名称（"hfl/chinese-pert-large-mrc"）和任务（"question-answering"）
    if not model_name_list:
        model_name_list = [
            "hfl/chinese-pert-large-mrc",
            "wptoux/albert-chinese-large-qa",
        ]
    for model_name in model_name_list:
        task = "question-answering"
        local_path = "./cache_folder/models/{0}".format(model_name.replace("/", "_"))
        if Path(local_path).exists():
            print(f"模型已存在本地路径: {local_path}")
            continue

        # 创建一个question-answering的pipeline
        qa_pipeline = pipeline(task, model=model_name, tokenizer=model_name)

        # 保存模型到本地路径

        qa_pipeline.model.save_pretrained(local_path)
        qa_pipeline.tokenizer.save_pretrained(local_path)

        print(f"模型已下载到本地路径: {local_path}")
    print("模型全部下载完成")


def modelscope_download(model_id):
    """
    从modelscope上面下载模型
    """
    # 将模型ID中的斜杠替换为下划线，以作为缓存目录的命名
    cache_dir = "./cache_folder/models"
    # 使用snapshot_download函数下载模型，并指定缓存目录
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    # 打印模型目录
    print(model_dir)


def load_chroma_db(embeddings):
    vector_store_path = get_vector_store_path()
    return Chroma(persist_directory=vector_store_path, embedding_function=embeddings)


def chat(query: str, content: str, history=None):
    model_id = "qwen/Qwen-1_8B-Chat-Int4"
    local_model_id = "./cache_folder/models/{0}".format(model_id)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=local_model_id,
                                              revision='master',
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_model_id,
        revision='master',
        device_map="cpu",
        trust_remote_code=True
    ).eval()

    prompt = """
    基于```内的内容回答问题。
```
{content}
```
我的问题是：{query}。
    """.format(query=query, content=content)
    return model.chat(tokenizer, prompt, history=history)
