#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - 无faiss依赖版本
使用numpy实现向量相似度计算，替代faiss索引
"""

import os
import json
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv

# 导入自定义模块
from embedding_module import EmbeddingAPIClient
from llm_api_client import GenerationAPIClient

class SimpleVectorIndex:
    """简单向量索引，使用numpy实现相似度搜索"""
    
    def __init__(self):
        self.vectors = None
        self.dimension = None
        self.ntotal = 0
    
    def add(self, vectors: np.ndarray):
        """添加向量到索引"""
        if self.vectors is None:
            self.vectors = vectors.copy()
            self.dimension = vectors.shape[1]
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.ntotal = self.vectors.shape[0]
    
    def search(self, query_vectors: np.ndarray, k: int = 5):
        """搜索最相似的向量"""
        if self.vectors is None or self.ntotal == 0:
            return np.array([]), np.array([])
        
        # 计算余弦相似度
        query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        vectors_norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        
        # 避免除零错误
        query_norm = np.where(query_norm == 0, 1, query_norm)
        vectors_norm = np.where(vectors_norm == 0, 1, vectors_norm)
        
        # 归一化向量
        query_normalized = query_vectors / query_norm
        vectors_normalized = self.vectors / vectors_norm
        
        # 计算相似度矩阵
        similarities = np.dot(query_normalized, vectors_normalized.T)
        
        # 转换为距离（1 - 相似度）
        distances = 1 - similarities
        
        # 获取top-k结果
        k = min(k, self.ntotal)
        indices = np.argsort(distances, axis=1)[:, :k]
        
        # 获取对应的距离
        batch_size = distances.shape[0]
        selected_distances = np.array([
            distances[i, indices[i]] for i in range(batch_size)
        ])
        
        return selected_distances, indices

class RAGPipelineNoFaiss:
    """RAG检索增强生成管道 - 无faiss依赖版本"""
    
    def __init__(self, index_file: str, metadata_file: str):
        """
        初始化RAG管道
        
        Args:
            index_file: 向量索引文件路径（.npy格式）
            metadata_file: 元数据文件路径（.json格式）
        """
        # 加载环境变量
        load_dotenv()
        
        # 初始化API客户端
        self.embedding_client = EmbeddingAPIClient()
        self.generation_client = GenerationAPIClient()
        
        # 加载向量索引和元数据
        self.index = SimpleVectorIndex()
        self.metadata = []
        
        self._load_index_and_metadata(index_file, metadata_file)
        
        print(f"✅ RAG管道初始化完成")
        print(f"📊 向量索引大小: {self.index.ntotal}")
        print(f"📋 元数据条目: {len(self.metadata)}")
    
    def _load_index_and_metadata(self, index_file: str, metadata_file: str):
        """加载向量索引和元数据"""
        try:
            # 加载向量数据
            if os.path.exists(index_file):
                vectors = np.load(index_file)
                self.index.add(vectors)
                print(f"✅ 向量索引加载成功: {vectors.shape}")
            else:
                raise FileNotFoundError(f"向量索引文件不存在: {index_file}")
            
            # 加载元数据
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✅ 元数据加载成功: {len(self.metadata)} 条记录")
            else:
                raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
            
            # 验证数据一致性
            if self.index.ntotal != len(self.metadata):
                raise ValueError(
                    f"向量数量({self.index.ntotal})与元数据数量({len(self.metadata)})不匹配"
                )
                
        except Exception as e:
            print(f"❌ 加载索引和元数据失败: {e}")
            raise
    
    def answer_question(self, question: str, top_k: int = 5, max_context_length: int = 3000) -> Dict[str, Any]:
        """
        回答问题的主要方法
        
        Args:
            question: 用户问题
            top_k: 检索的文档数量
            max_context_length: 最大上下文长度
            
        Returns:
            包含答案、来源和相关信息的字典
        """
        try:
            print(f"\n🔍 开始处理问题: {question}")
            
            # 1. 将问题转换为向量
            print("📝 正在生成问题向量...")
            question_vector = self.embedding_client.get_embeddings([question])
            question_vector = np.array(question_vector).reshape(1, -1)
            
            # 2. 在向量索引中搜索相似文档
            print(f"🔎 正在检索相关文档 (top_k={top_k})...")
            distances, indices = self.index.search(question_vector, k=top_k)
            
            # 3. 获取检索到的文档
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                doc_info = self.metadata[idx].copy()
                doc_info['similarity_score'] = float(1 - distances[0][i])  # 转换回相似度
                doc_info['rank'] = i + 1
                retrieved_docs.append(doc_info)
            
            print(f"📚 检索到 {len(retrieved_docs)} 个相关文档")
            
            # 4. 构建上下文
            context = self._build_context(retrieved_docs, max_context_length)
            
            # 5. 生成答案
            print("🤖 正在生成答案...")
            answer = self._generate_answer(question, context, retrieved_docs)
            
            # 6. 构建返回结果
            result = {
                'question': question,
                'answer': answer,
                'sources': retrieved_docs,
                'context_used': context,
                'retrieval_stats': {
                    'total_docs_in_index': self.index.ntotal,
                    'docs_retrieved': len(retrieved_docs),
                    'top_similarity': float(retrieved_docs[0]['similarity_score']) if retrieved_docs else 0.0
                }
            }
            
            print("✅ 问题处理完成")
            return result
            
        except Exception as e:
            print(f"❌ 处理问题时出错: {e}")
            return {
                'question': question,
                'answer': f"抱歉，处理您的问题时出现错误: {str(e)}",
                'sources': [],
                'error': str(e)
            }
    
    def _build_context(self, retrieved_docs: List[Dict], max_length: int) -> str:
        """构建上下文字符串"""
        context_parts = []
        current_length = 0
        
        for doc in retrieved_docs:
            content = doc.get('content', '')
            source_info = f"[来源: {doc.get('source', '未知')}]"
            
            part = f"{source_info}\n{content}\n"
            
            if current_length + len(part) > max_length:
                break
                
            context_parts.append(part)
            current_length += len(part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """生成答案"""
        # 构建提示词
        source_list = "\n".join([
            f"{i+1}. {doc.get('source', '未知来源')} (相似度: {doc.get('similarity_score', 0):.3f})"
            for i, doc in enumerate(sources[:3])
        ])
        
        prompt = f"""请基于以下上下文信息回答问题。要求：
1. 答案必须基于提供的上下文信息
2. 如果上下文中没有相关信息，请明确说明
3. 引用具体的来源信息
4. 保持答案准确、简洁

上下文信息：
{context}

主要来源：
{source_list}

问题：{question}

请提供详细的答案："""
        
        try:
            response = self.generation_client.generate(prompt)
            return response
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        return {
            'index_size': self.index.ntotal,
            'metadata_count': len(self.metadata),
            'vector_dimension': self.index.dimension,
            'embedding_client_ready': hasattr(self.embedding_client, 'base_url'),
            'generation_client_ready': hasattr(self.generation_client, 'base_url')
        }

# 使用示例
if __name__ == "__main__":
    # 示例用法
    index_file = "output/document_vectors.npy"
    metadata_file = "output/document_metadata.json"
    
    try:
        # 初始化RAG管道
        rag = RAGPipelineNoFaiss(index_file, metadata_file)
        
        # 测试问题
        test_question = "什么是人工智能？"
        result = rag.answer_question(test_question)
        
        print("\n" + "="*50)
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"来源数量: {len(result['sources'])}")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")