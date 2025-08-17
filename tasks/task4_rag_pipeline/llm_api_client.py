import os
import requests
import json
from typing import Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

class GenerationAPIClient:
    """
    通用文本生成API客户端，支持与OpenAI v1/chat/completions兼容的API服务
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
        self.model_name = os.getenv('LOCAL_TEXT_MODEL')
        
        # 配置校验
        if not self.base_url:
            raise ValueError("LOCAL_BASE_URL 环境变量未设置，请检查.env文件")
        if not self.model_name:
            raise ValueError("LOCAL_TEXT_MODEL 环境变量未设置，请检查.env文件")
        
        # 构建完整的端点URL
        self.endpoint_url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        
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
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        生成文本回答
        
        Args:
            prompt: 输入的提示词
            temperature: 生成温度，控制随机性
            max_tokens: 最大生成token数
            
        Returns:
            生成的文本内容
            
        Raises:
            requests.RequestException: API请求失败
            ValueError: 响应格式错误
        """
        if not prompt:
            return ""
        
        # 构建请求体
        payload = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        try:
            # 发送API请求
            response = self.session.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 提取生成的文本
            if 'choices' not in result or not result['choices']:
                raise ValueError(f"API响应格式错误：缺少'choices'字段。响应内容：{result}")
            
            choice = result['choices'][0]
            if 'message' not in choice or 'content' not in choice['message']:
                raise ValueError(f"API响应格式错误：缺少消息内容。选择内容：{choice}")
            
            return choice['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            raise
        except (KeyError, IndexError, TypeError) as e:
            print(f"响应解析失败: {e}")
            raise ValueError(f"API响应格式错误: {e}")
    
    def test_connection(self) -> bool:
        """
        测试API连接是否正常
        
        Returns:
            连接是否成功
        """
        try:
            test_response = self.generate("Hello", max_tokens=10)
            return len(test_response) > 0
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False
    
    def __del__(self):
        """
        清理资源
        """
        if hasattr(self, 'session'):
            self.session.close()

# 测试代码
if __name__ == "__main__":
    try:
        client = GenerationAPIClient()
        print(f"初始化成功，使用模型: {client.model_name}")
        print(f"端点URL: {client.endpoint_url}")
        
        # 测试连接
        if client.test_connection():
            print("✓ API连接测试成功")
            
            # 测试生成
            test_prompt = "请简单介绍一下人工智能。"
            response = client.generate(test_prompt, max_tokens=100)
            print(f"\n测试生成结果:\n{response}")
        else:
            print("✗ API连接测试失败")
            
    except Exception as e:
        print(f"初始化失败: {e}")