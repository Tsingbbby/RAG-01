#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化模块 - 使用BAAI/bge-m3模型进行文本向量化

该模块提供了一个统一的接口来加载和使用bge-m3模型，
支持批量处理以优化GPU利用率。
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    BAAI/bge-m3 向量化模型封装类
    
    该类负责加载bge-m3模型并提供批量向量化功能。
    模型会自动检测并使用可用的GPU设备。
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        """
        初始化向量化模型
        
        Args:
            model_name: 模型名称，默认为BAAI/bge-m3
            batch_size: 批处理大小，默认为32
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 检测设备
        self.device = self._detect_device()
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和tokenizer
        logger.info(f"正在加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        logger.info("模型加载完成")
    
    def _detect_device(self) -> str:
        """
        检测可用的计算设备
        
        Returns:
            设备字符串 ('cuda' 或 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon GPU
            return "mps"
        else:
            return "cpu"
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量
        
        Args:
            texts: 待向量化的文本列表
            
        Returns:
            向量列表，每个向量为float列表
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"已处理 {i + len(batch_texts)}/{len(texts)} 个文本")
        
        logger.info(f"向量化完成，共生成 {len(all_embeddings)} 个向量")
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        处理单个批次的文本向量化
        
        Args:
            texts: 批次文本列表
            
        Returns:
            批次向量列表
        """
        with torch.no_grad():
            # 编码文本
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移动到设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 获取模型输出
            outputs = self.model(**encoded)
            
            # 使用CLS token的输出作为句子向量
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # 转换为CPU并转为列表
            embeddings = embeddings.cpu().numpy().tolist()
            
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            向量维度
        """
        return self.model.config.hidden_size

# 测试代码
if __name__ == "__main__":
    # 简单测试
    embedding_model = EmbeddingModel()
    
    test_texts = [
        "这是一个测试文本。",
        "This is another test sentence.",
        "向量化模型测试中..."
    ]
    
    embeddings = embedding_model.embed(test_texts)
    
    print(f"生成了 {len(embeddings)} 个向量")
    print(f"向量维度: {len(embeddings[0])}")
    print(f"第一个向量的前5个值: {embeddings[0][:5]}")