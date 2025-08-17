import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingAPIClient:
    """
    通用向量化API客户端，支持与OpenAI v1/embeddings兼容的API服务
    包括硅基流动、Xinference等服务
    """
    
    def __init__(self):
        """
        初始化API客户端，从.env文件加载配置
        """
        # 加载环境变量
        load_dotenv()
        
        # 读取配置
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.model_name = os.getenv('LOCAL_EMBEDDING_MODEL')
        
        # 配置校验
        if not self.base_url:
            raise ValueError("LOCAL_BASE_URL 环境变量未设置，请检查.env文件")
        if not self.model_name:
            raise ValueError("LOCAL_EMBEDDING_MODEL 环境变量未设置，请检查.env文件")
        
        # 构建完整的端点URL
        self.endpoint_url = f"{self.base_url.rstrip('/')}/v1/embeddings"
        
        # 初始化可复用的session
        self.session = requests.Session()
        
        # 设置请求头
        self.headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        对文本列表进行向量化
        
        Args:
            texts: 待向量化的文本列表
            
        Returns:
            向量列表，每个向量为float列表
            
        Raises:
            requests.RequestException: API请求失败
            ValueError: 响应格式错误
        """
        if not texts:
            return []
        
        # 构建请求体
        payload = {
            'input': texts,
            'model': self.model_name
        }
        
        try:
            # 发送API请求
            response = self.session.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 提取向量数据
            if 'data' not in result:
                raise ValueError(f"API响应格式错误：缺少'data'字段。响应内容：{result}")
            
            embeddings = []
            for item in result['data']:
                if 'embedding' not in item:
                    raise ValueError(f"API响应格式错误：缺少'embedding'字段。项目内容：{item}")
                embeddings.append(item['embedding'])
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"响应解析失败: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        获取向量维度（通过发送单个测试文本）
        
        Returns:
            向量维度
        """
        test_embedding = self.embed(["test"])
        if test_embedding:
            return len(test_embedding[0])
        return 0
    
    def __del__(self):
        """
        清理资源
        """
        if hasattr(self, 'session'):
            self.session.close()