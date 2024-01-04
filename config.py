import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    配置文件
    """

    embedding_model_name = 'BAAI/bge-small-zh-v1.5'  # 嵌入模型名称
    pipeline_model_names = ['hfl/chinese-pert-large-mrc', 'wptoux/albert-chinese-large-qa']  # 流水线模型名称列表
    vector_store_path = 'vector_store_path'  # 向量存储路径
    dashscope_api_key = os.getenv(key='DASHSCOPE_API_KEY', default=None)  # Dashscope API Key
