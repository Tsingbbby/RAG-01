import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
import faiss
from dotenv import load_dotenv
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 导入基础RAG管道
from rag_pipeline import RAGPipeline

class EnhancedRAGPipeline(RAGPipeline):
    """
    增强版RAG管道，继承自RAGPipeline
    集成重排模型以提升检索精度
    """
    
    def __init__(self, 
                 index_path: str = "output/knowledge_base.index",
                 metadata_path: str = "output/chunk_metadata.pkl",
                 initial_k: int = 20,  # 初始检索数量
                 final_k: int = 5,     # 重排后最终返回数量
                 enable_reranking: bool = True):
        """
        初始化增强版RAG管道
        
        Args:
            index_path: FAISS索引文件路径
            metadata_path: 元数据文件路径
            initial_k: 初始检索返回的文档数量
            final_k: 重排序后最终返回的文档数量
            enable_reranking: 是否启用重排序
        """
        # 调用父类初始化，使用initial_k作为基础k值
        super().__init__(index_path, metadata_path, initial_k)
        
        # 增强版特有参数
        self.initial_k = initial_k
        self.final_k = final_k
        self.enable_reranking = enable_reranking
        
        # 性能监控
        self.performance_metrics = []
        
        print(f"增强版RAG管道配置: 初始检索={initial_k}, 最终返回={final_k}, 重排序={'启用' if enable_reranking else '禁用'}")
        
        # 初始化TF-IDF重排序器
        if self.enable_reranking:
            self._initialize_tfidf_reranker()
        
        print("增强版RAG管道初始化完成！\n")
    
    def _initialize_tfidf_reranker(self):
        """初始化TF-IDF重排序器"""
        print("正在初始化TF-IDF重排序器...")
        
        try:
            # 提取所有文档内容用于训练TF-IDF
            all_texts = []
            for chunk_info in self.metadata:
                text = chunk_info.get('text', '')
                if text:
                    all_texts.append(text)
            
            if not all_texts:
                print("⚠️ 警告: 没有找到文档文本，禁用重排序功能")
                self.enable_reranking = False
                return
            
            # 初始化TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # 保留中文支持
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # 训练TF-IDF模型
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            print(f"✓ TF-IDF重排序器初始化成功，特征维度: {self.tfidf_matrix.shape[1]}")
            
        except Exception as e:
            print(f"⚠️ TF-IDF重排序器初始化失败: {e}，禁用重排序功能")
            self.enable_reranking = False
    
    def _rerank_documents(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        使用TF-IDF和向量相似度加权的混合重排序算法
        
        Args:
            query: 用户查询
            retrieved_docs: 初始检索到的文档列表
            
        Returns:
            重排序后的文档列表
        """
        if not self.enable_reranking or not hasattr(self, 'tfidf_vectorizer'):
            return retrieved_docs[:self.final_k]
        
        try:
            # 向量化查询
            query_tfidf = self.tfidf_vectorizer.transform([query])
            
            # 计算查询与每个候选文档的TF-IDF相似度
            rerank_scores = []
            for doc in retrieved_docs:
                doc_idx = doc['index']
                if doc_idx < self.tfidf_matrix.shape[0]:
                    # 获取文档的TF-IDF向量
                    doc_tfidf = self.tfidf_matrix[doc_idx:doc_idx+1]
                    tfidf_sim = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
                    
                    # 将FAISS距离转换为相似度分数
                    vector_sim = 1.0 / (1.0 + doc['distance'])
                    
                    # 加权融合：70%向量相似度 + 30%TF-IDF相似度
                    combined_score = 0.7 * vector_sim + 0.3 * tfidf_sim
                    
                    rerank_scores.append((doc, combined_score))
                else:
                    # 如果索引超出范围，使用原始分数
                    vector_sim = 1.0 / (1.0 + doc['distance'])
                    rerank_scores.append((doc, vector_sim))
            
            # 按组合分数降序排序
            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, score in rerank_scores]
            
            return reranked_docs[:self.final_k]
            
        except Exception as e:
            print(f"⚠️ 重排序失败: {e}，使用原始排序")
            return retrieved_docs[:self.final_k]
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        重写父类的answer_question方法，集成重排序功能
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案、来源信息和性能指标的字典
        """
        start_time = time.time()
        
        try:
            # Step 1: 查询向量化
            embedding_start = time.time()
            try:
                query_embedding = self.embedding_client.get_embedding(question)
                if query_embedding is None:
                    raise ValueError("查询向量化失败")
                query_vector = np.array([query_embedding], dtype=np.float32)
            except Exception as e:
                return {
                    'answer': f"查询向量化失败: {e}",
                    'filename': 'N/A',
                    'page': 'N/A',
                    'sources': [],
                    'performance_metrics': {
                        'total_time': time.time() - start_time,
                        'embedding_time': 0,
                        'retrieval_time': 0,
                        'reranking_time': 0,
                        'generation_time': 0
                    }
                }
            embedding_time = time.time() - embedding_start
            
            # Step 2: 向量检索（使用更大的initial_k）
            retrieval_start = time.time()
            try:
                distances, indices = self.index.search(query_vector, self.initial_k)
                
                retrieved_docs = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx != -1 and idx < len(self.metadata):
                        doc_info = self.metadata[idx].copy()
                        doc_info['distance'] = float(distance)
                        doc_info['index'] = int(idx)
                        retrieved_docs.append(doc_info)
                
                if not retrieved_docs:
                    raise ValueError("未找到相关文档")
                    
            except Exception as e:
                return {
                    'answer': f"文档检索失败: {e}",
                    'filename': 'N/A',
                    'page': 'N/A',
                    'sources': [],
                    'performance_metrics': {
                        'total_time': time.time() - start_time,
                        'embedding_time': embedding_time,
                        'retrieval_time': 0,
                        'reranking_time': 0,
                        'generation_time': 0
                    }
                }
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: 重排序（如果启用）
            reranking_start = time.time()
            if self.enable_reranking:
                final_docs = self._rerank_documents(question, retrieved_docs)
            else:
                final_docs = retrieved_docs[:self.final_k]
            reranking_time = time.time() - reranking_start
            
            # Step 4: 构建上下文
            context_parts = []
            for doc in final_docs:
                text = doc.get('text', '').strip()
                if text:
                    filename = doc.get('filename', 'unknown')
                    page = doc.get('page', 'unknown')
                    context_parts.append(f"[来源: {filename}, 第{page}页]\n{text}")
            
            if not context_parts:
                return {
                    'answer': "未找到相关的上下文信息",
                    'filename': 'N/A',
                    'page': 'N/A',
                    'sources': final_docs,
                    'performance_metrics': {
                        'total_time': time.time() - start_time,
                        'embedding_time': embedding_time,
                        'retrieval_time': retrieval_time,
                        'reranking_time': reranking_time,
                        'generation_time': 0
                    }
                }
            
            context = "\n\n".join(context_parts)
            
            # Step 5: 构建提示词并生成答案
            generation_start = time.time()
            try:
                prompt = self._build_prompt(question, context)
                answer = self.generation_client.generate_response(prompt)
                
                if not answer or answer.strip() == "":
                    answer = "根据提供的信息，无法回答该问题。"
                    
            except Exception as e:
                answer = f"答案生成失败: {e}"
            generation_time = time.time() - generation_start
            
            # 记录性能指标
            total_time = time.time() - start_time
            performance_metrics = {
                'total_time': total_time,
                'embedding_time': embedding_time,
                'retrieval_time': retrieval_time,
                'reranking_time': reranking_time,
                'generation_time': generation_time,
                'initial_retrieved': len(retrieved_docs),
                'final_retrieved': len(final_docs)
            }
            
            self.performance_metrics.append(performance_metrics)
            
            # 返回结果
            main_source = final_docs[0] if final_docs else {'filename': 'N/A', 'page': 'N/A'}
            
            return {
                'answer': answer,
                'filename': main_source.get('filename', 'N/A'),
                'page': main_source.get('page', 'N/A'),
                'sources': final_docs,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            return {
                'answer': f"处理问题时发生错误: {e}",
                'filename': 'N/A',
                'page': 'N/A',
                'sources': [],
                'performance_metrics': {
                    'total_time': time.time() - start_time,
                    'embedding_time': 0,
                    'retrieval_time': 0,
                    'reranking_time': 0,
                    'generation_time': 0
                }
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            包含各项性能指标统计的字典
        """
        if not self.performance_metrics:
            return {"message": "暂无性能数据"}
        
        # 计算平均值
        avg_metrics = {}
        for key in ['total_time', 'embedding_time', 'retrieval_time', 'reranking_time', 'generation_time']:
            values = [m[key] for m in self.performance_metrics]
            avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return {
            'total_queries': len(self.performance_metrics),
            'configuration': {
                'initial_k': self.initial_k,
                'final_k': self.final_k,
                'reranking_enabled': self.enable_reranking
            },
            **avg_metrics
        }

if __name__ == "__main__":
    try:
        print("测试增强版RAG管道...")
        
        # 初始化增强版RAG管道
        enhanced_rag = EnhancedRAGPipeline(
            initial_k=20,  # 初始检索更多文档
            final_k=5,     # 最终返回5个
            enable_reranking=True
        )
        
        # 获取管道信息
        info = enhanced_rag.get_pipeline_info()
        print("增强版管道配置信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # 测试问题
        test_questions = [
            "公司在2023年的重大产品临床进展是什么？",
            "公司的主要业务发展策略如何？",
            "与竞争对手相比，公司有哪些优势？"
        ]
        
        for question in test_questions:
            print(f"\n{'='*80}")
            result = enhanced_rag.answer_question(question)
            
            print(f"\n问题: {question}")
            print(f"答案: {result['answer']}")
            print(f"主要来源: {result['filename']} (第{result['page']}页)")
            metrics = result['performance_metrics']
            print(f"性能指标: 总耗时 {metrics['total_time']:.2f}s (重排序: {metrics['reranking_time']:.3f}s)")
            print(f"检索数量: 初始{metrics['initial_retrieved']} -> 最终{metrics['final_retrieved']}")
            print(f"{'='*80}")
        
        # 显示性能报告
        print("\n" + "="*80)
        print("性能报告:")
        report = enhanced_rag.get_performance_report()
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"增强版RAG管道测试失败: {e}")
        import traceback
        traceback.print_exc()