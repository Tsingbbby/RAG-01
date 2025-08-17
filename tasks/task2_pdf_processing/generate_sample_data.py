#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成示例数据和向量化文件
用于演示RAG管道功能
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from embedding_module import EmbeddingAPIClient

class SampleDataGenerator:
    """示例数据生成器"""
    
    def __init__(self):
        self.embedding_client = EmbeddingAPIClient()
        self.output_dir = "output"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_sample_documents(self) -> List[Dict[str, Any]]:
        """生成示例文档数据"""
        documents = [
            {
                "id": "doc_001",
                "title": "人工智能基础",
                "content": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能的研究领域包括机器学习、深度学习、自然语言处理、计算机视觉等。",
                "source": "AI教程第1章",
                "metadata": {"category": "基础概念", "difficulty": "初级"}
            },
            {
                "id": "doc_002",
                "title": "机器学习概述",
                "content": "机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。机器学习算法通过训练数据来构建数学模型，以便对新数据做出预测或决策。主要的机器学习方法包括监督学习、无监督学习和强化学习。",
                "source": "机器学习入门指南",
                "metadata": {"category": "机器学习", "difficulty": "中级"}
            },
            {
                "id": "doc_003",
                "title": "深度学习原理",
                "content": "深度学习是机器学习的一个子集，它基于人工神经网络，特别是深层神经网络。深度学习模型能够自动学习数据的层次化表示，在图像识别、语音识别、自然语言处理等领域取得了突破性进展。常见的深度学习架构包括卷积神经网络（CNN）、循环神经网络（RNN）和Transformer。",
                "source": "深度学习实战",
                "metadata": {"category": "深度学习", "difficulty": "高级"}
            },
            {
                "id": "doc_004",
                "title": "自然语言处理技术",
                "content": "自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。NLP的主要任务包括文本分类、情感分析、命名实体识别、机器翻译、问答系统等。",
                "source": "NLP技术手册",
                "metadata": {"category": "自然语言处理", "difficulty": "中级"}
            },
            {
                "id": "doc_005",
                "title": "计算机视觉应用",
                "content": "计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理。计算机视觉的应用领域包括图像识别、目标检测、人脸识别、医学影像分析等。",
                "source": "计算机视觉导论",
                "metadata": {"category": "计算机视觉", "difficulty": "中级"}
            },
            {
                "id": "doc_006",
                "title": "强化学习基础",
                "content": "强化学习是机器学习的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。强化学习的特点是试错搜索和延迟回报。在强化学习中，智能体通过与环境的交互来学习最优策略。著名的强化学习算法包括Q-learning、策略梯度方法和Actor-Critic方法。",
                "source": "强化学习原理与实践",
                "metadata": {"category": "强化学习", "difficulty": "高级"}
            },
            {
                "id": "doc_007",
                "title": "数据预处理技术",
                "content": "数据预处理是机器学习和数据挖掘过程中的重要步骤。它包括数据清洗、数据集成、数据变换和数据规约等操作。良好的数据预处理能够提高模型的性能和准确性。常见的预处理技术包括缺失值处理、异常值检测、特征缩放、特征选择等。",
                "source": "数据科学实战",
                "metadata": {"category": "数据处理", "difficulty": "初级"}
            },
            {
                "id": "doc_008",
                "title": "模型评估方法",
                "content": "模型评估是机器学习项目中的关键环节，用于衡量模型的性能和泛化能力。常用的评估方法包括交叉验证、留出法、自助法等。评估指标根据任务类型而不同，分类任务常用准确率、精确率、召回率、F1分数，回归任务常用均方误差、平均绝对误差等。",
                "source": "机器学习评估指南",
                "metadata": {"category": "模型评估", "difficulty": "中级"}
            }
        ]
        
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 200) -> List[Dict[str, Any]]:
        """将文档分块"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            
            # 简单的分块策略：按字符长度分块
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                
                chunk = {
                    "chunk_id": f"{doc['id']}_chunk_{i//chunk_size + 1}",
                    "document_id": doc["id"],
                    "title": doc["title"],
                    "content": chunk_content,
                    "source": doc["source"],
                    "metadata": doc["metadata"].copy(),
                    "chunk_index": i // chunk_size + 1,
                    "start_char": i,
                    "end_char": min(i + chunk_size, len(content))
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为文档块生成向量嵌入"""
        vectorized_chunks = []
        
        print(f"正在为 {len(chunks)} 个文档块生成向量嵌入...")
        
        for i, chunk in enumerate(chunks):
            try:
                # 生成向量嵌入
                embedding = self.embedding_client.get_embedding(chunk["content"])
                
                # 创建向量化数据
                vectorized_chunk = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "metadata": chunk["metadata"],
                    "chunk_index": chunk["chunk_index"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "embedding": embedding,
                    "embedding_dim": len(embedding)
                }
                
                vectorized_chunks.append(vectorized_chunk)
                
                print(f"✓ 已处理 {i+1}/{len(chunks)} 个块")
                
            except Exception as e:
                print(f"❌ 处理块 {chunk['chunk_id']} 时出错: {e}")
                continue
        
        return vectorized_chunks
    
    def save_vectorized_data(self, vectorized_chunks: List[Dict[str, Any]]):
        """保存向量化数据到文件"""
        output_file = os.path.join(self.output_dir, "vectorized_chunks.jsonl")
        
        print(f"正在保存向量化数据到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in vectorized_chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ 已保存 {len(vectorized_chunks)} 个向量化块到 {output_file}")
    
    def generate_all(self):
        """生成完整的示例数据集"""
        print("=== 生成示例数据和向量化文件 ===")
        
        # 1. 生成示例文档
        print("\n1. 生成示例文档...")
        documents = self.generate_sample_documents()
        print(f"✓ 生成了 {len(documents)} 个示例文档")
        
        # 2. 文档分块
        print("\n2. 文档分块处理...")
        chunks = self.chunk_documents(documents, chunk_size=150)
        print(f"✓ 生成了 {len(chunks)} 个文档块")
        
        # 3. 生成向量嵌入
        print("\n3. 生成向量嵌入...")
        vectorized_chunks = self.generate_embeddings(chunks)
        print(f"✓ 成功生成 {len(vectorized_chunks)} 个向量化块")
        
        # 4. 保存数据
        print("\n4. 保存向量化数据...")
        self.save_vectorized_data(vectorized_chunks)
        
        print("\n🎉 示例数据生成完成！")
        print(f"📁 输出文件: {os.path.join(self.output_dir, 'vectorized_chunks.jsonl')}")
        
        return vectorized_chunks

def main():
    """主函数"""
    try:
        generator = SampleDataGenerator()
        generator.generate_all()
        
    except Exception as e:
        print(f"❌ 生成过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()