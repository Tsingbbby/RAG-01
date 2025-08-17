#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG管道演示脚本 - 使用FAISS
完整的RAG管道演示，包括数据生成、索引构建和问答功能
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_module import EmbeddingAPIClient
from llm_api_client import GenerationAPIClient
from rag_pipeline import RAGPipeline

def generate_sample_documents() -> List[Dict[str, Any]]:
    """
    生成示例文档数据
    """
    return [
        {
            "title": "强化学习基础",
            "content": "强化学习是机器学习的一个重要分支，通过智能体与环境的交互来学习最优策略。强化学习在游戏AI、机器人控制、自动驾驶、推荐系统等领域有广泛应用。AlphaGo就是强化学习在围棋领域的成功应用案例。",
            "source": "ml_textbook.pdf"
        },
        {
            "title": "强化学习应用领域",
            "content": "强化学习的主要应用包括：1）游戏AI：如AlphaGo、Dota2 AI等；2）机器人控制：机械臂操作、无人机导航；3）自动驾驶：路径规划、决策控制；4）金融交易：算法交易、风险管理；5）推荐系统：个性化推荐、广告投放；6）资源调度：云计算资源分配、网络流量优化。",
            "source": "ai_applications.pdf"
        },
        {
            "title": "深度强化学习",
            "content": "深度强化学习结合了深度学习和强化学习，使用神经网络来近似价值函数或策略函数。DQN、A3C、PPO等算法在Atari游戏、机器人控制等任务中取得了突破性进展。",
            "source": "deep_rl.pdf"
        },
        {
            "title": "自然语言处理概述",
            "content": "自然语言处理（NLP）是人工智能的重要分支，涉及文本分析、语言理解、机器翻译、情感分析等任务。现代NLP广泛使用Transformer架构和预训练语言模型如BERT、GPT等。",
            "source": "nlp_guide.pdf"
        },
        {
            "title": "机器学习算法",
            "content": "机器学习包括监督学习、无监督学习和强化学习三大类。监督学习用于分类和回归任务，无监督学习用于聚类和降维，强化学习用于序列决策问题。每种方法都有其特定的应用场景和优势。",
            "source": "ml_algorithms.pdf"
        },
        {
            "title": "AI在医疗中的应用",
            "content": "人工智能在医疗领域的应用包括医学影像诊断、药物发现、个性化治疗、疾病预测等。强化学习在治疗方案优化、药物剂量调节等方面也有重要应用。",
            "source": "ai_healthcare.pdf"
        }
    ]

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    将文档分割成小块
    """
    chunks = []
    for doc in documents:
        content = doc['content']
        # 简单的分块策略：按字符数分割
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            if len(chunk_content.strip()) > 0:
                chunks.append({
                    'content': chunk_content,
                    'title': doc['title'],
                    'source': doc['source'],
                    'chunk_id': len(chunks)
                })
    return chunks

def generate_vectorized_data():
    """
    生成向量化数据文件
    """
    print("=== 生成示例数据 ===")
    
    # 生成文档和分块
    documents = generate_sample_documents()
    chunks = chunk_documents(documents)
    
    print(f"生成了 {len(documents)} 个文档，{len(chunks)} 个文本块")
    
    # 初始化嵌入客户端
    embedding_client = EmbeddingAPIClient()
    
    # 生成向量化数据
    vectorized_chunks = []
    print("正在生成向量嵌入...")
    
    for i, chunk in enumerate(chunks):
        try:
            # 获取嵌入向量
            embedding = embedding_client.get_embedding(chunk['content'])
            
            vectorized_chunk = {
                'content': chunk['content'],
                'title': chunk['title'],
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'embedding': embedding
            }
            vectorized_chunks.append(vectorized_chunk)
            
            if (i + 1) % 2 == 0:
                print(f"已处理 {i + 1}/{len(chunks)} 个文本块")
                
        except Exception as e:
            print(f"处理第 {i+1} 个文本块时出错: {e}")
            continue
    
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    
    # 保存向量化数据
    output_file = 'output/vectorized_chunks.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in vectorized_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"向量化数据已保存到 {output_file}")
    print(f"共生成 {len(vectorized_chunks)} 个向量化文本块")
    
    return len(vectorized_chunks)

def build_faiss_index():
    """
    构建FAISS索引
    """
    print("\n=== 构建FAISS索引 ===")
    
    try:
        import faiss
    except ImportError:
        print("错误：未安装faiss库，请运行: pip install faiss-cpu")
        return False
    
    # 读取向量化数据
    vectorized_file = 'output/vectorized_chunks.jsonl'
    if not os.path.exists(vectorized_file):
        print(f"错误：找不到向量化数据文件 {vectorized_file}")
        return False
    
    chunks = []
    embeddings = []
    
    with open(vectorized_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
            embeddings.append(chunk['embedding'])
    
    print(f"加载了 {len(chunks)} 个向量化文本块")
    
    # 转换为numpy数组
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    
    print(f"向量维度: {dimension}")
    
    # 创建FAISS索引
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
    
    # 归一化向量（用于余弦相似度）
    faiss.normalize_L2(embeddings_array)
    
    # 添加向量到索引
    index.add(embeddings_array)
    
    print(f"FAISS索引构建完成，包含 {index.ntotal} 个向量")
    
    # 保存索引
    index_file = 'output/knowledge_base.index'
    faiss.write_index(index, index_file)
    print(f"FAISS索引已保存到 {index_file}")
    
    # 保存元数据
    metadata = []
    for chunk in chunks:
        metadata.append({
            'content': chunk['content'],
            'title': chunk['title'],
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id']
        })
    
    metadata_file = 'output/chunk_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"元数据已保存到 {metadata_file}")
    
    return True

def test_rag_pipeline():
    """
    测试RAG管道
    """
    print("\n=== 测试RAG管道 ===")
    
    # 检查必需文件
    required_files = [
        'output/knowledge_base.index',
        'output/chunk_metadata.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误：找不到必需文件 {file_path}")
            return False
    
    try:
        # 初始化RAG管道
        rag = RAGPipeline(
            index_path='output/knowledge_base.index',
            metadata_path='output/chunk_metadata.pkl'
        )
        
        print("RAG管道初始化成功")
        
        # 测试问题
        test_questions = [
            "什么是人工智能？",
            "机器学习有哪些类型？",
            "深度学习网络有什么特点？",
            "自然语言处理包括哪些任务？",
            "强化学习有什么应用？"
        ]
        
        print("\n开始问答测试：")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 30)
            
            try:
                result = rag.answer_question(question)
                print(f"回答: {result['answer']}")
                if 'sources' in result:
                    print(f"来源: {result['sources']}")
            except Exception as e:
                print(f"回答问题时出错: {e}")
        
        return True
        
    except Exception as e:
        print(f"RAG管道测试失败: {e}")
        return False

def interactive_qa():
    """
    交互式问答
    """
    print("\n=== 交互式问答 ===")
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
    
    try:
        # 初始化RAG管道
        rag = RAGPipeline(
            index_path='output/knowledge_base.index',
            metadata_path='output/chunk_metadata.pkl'
        )
        
        while True:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用！")
                break
            
            if not question:
                continue
            
            try:
                print("正在思考...")
                result = rag.answer_question(question)
                print(f"\n回答: {result['answer']}")
                if 'sources' in result:
                    print(f"来源: {result['sources']}")
            except Exception as e:
                print(f"回答问题时出错: {e}")
    
    except Exception as e:
        print(f"初始化RAG管道失败: {e}")

def main():
    """
    主函数
    """
    print("RAG管道演示 - 使用FAISS")
    print("=" * 40)
    
    # 检查环境变量
    required_env_vars = ['LOCAL_API_KEY', 'LOCAL_BASE_URL', 'LOCAL_EMBEDDING_MODEL', 'LOCAL_TEXT_MODEL']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"错误：缺少环境变量: {', '.join(missing_vars)}")
        print("请检查.env文件")
        return
    
    try:
        # 步骤1: 生成向量化数据
        if not os.path.exists('output/vectorized_chunks.jsonl'):
            chunk_count = generate_vectorized_data()
            if chunk_count == 0:
                print("生成向量化数据失败")
                return
        else:
            print("向量化数据文件已存在，跳过生成步骤")
        
        # 步骤2: 构建FAISS索引
        if not os.path.exists('output/knowledge_base.index'):
            if not build_faiss_index():
                print("构建FAISS索引失败")
                return
        else:
            print("FAISS索引文件已存在，跳过构建步骤")
        
        # 步骤3: 测试RAG管道
        if not test_rag_pipeline():
            print("RAG管道测试失败")
            return
        
        # 步骤4: 交互式问答
        interactive_qa()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()