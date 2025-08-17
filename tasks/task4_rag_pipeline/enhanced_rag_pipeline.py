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

# 导入自定义模块
from embedding_module import EmbeddingAPIClient
from llm_api_client import GenerationAPIClient

class EnhancedRAGPipeline:
    """
    增强版RAG管道，集成多种优化技术提升检索准确率
    """
    
    def __init__(self, 
                 index_path: str = "output/knowledge_base.index",
                 metadata_path: str = "output/chunk_metadata.pkl",
                 k: int = 10,  # 增加初始检索数量
                 final_k: int = 5,  # 最终返回数量
                 enable_reranking: bool = True,
                 enable_query_expansion: bool = True):
        """
        初始化增强版RAG管道
        
        Args:
            index_path: FAISS索引文件路径
            metadata_path: 元数据文件路径
            k: 初始检索返回的文档数量
            final_k: 重排序后最终返回的文档数量
            enable_reranking: 是否启用重排序
            enable_query_expansion: 是否启用查询扩展
        """
        self.k = k
        self.final_k = final_k
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # 性能监控
        self.performance_metrics = []
        
        # 加载环境变量
        load_dotenv()
        
        print("正在初始化增强版RAG管道...")
        
        # 初始化API客户端
        try:
            self.embedding_client = EmbeddingAPIClient()
            print("✓ Embedding API客户端初始化成功")
        except Exception as e:
            raise RuntimeError(f"Embedding API客户端初始化失败: {e}")
        
        try:
            self.generation_client = GenerationAPIClient()
            print("✓ Generation API客户端初始化成功")
        except Exception as e:
            raise RuntimeError(f"Generation API客户端初始化失败: {e}")
        
        # 加载FAISS索引
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            print(f"✓ FAISS索引加载成功，维度: {self.index.d}, 向量数量: {self.index.ntotal}")
        except Exception as e:
            raise RuntimeError(f"FAISS索引加载失败: {e}")
        
        # 加载元数据
        try:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✓ 元数据加载成功，共 {len(self.metadata)} 个文档块")
        except Exception as e:
            raise RuntimeError(f"元数据加载失败: {e}")
        
        # 验证数据一致性
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(f"数据不一致：元数据数量({len(self.metadata)}) != 索引向量数量({self.index.ntotal})")
        
        # 初始化TF-IDF向量化器用于重排序
        if self.enable_reranking:
            self._initialize_tfidf_reranker()
        
        print("增强版RAG管道初始化完成！\n")
    
    def _initialize_tfidf_reranker(self):
        """初始化TF-IDF重排序器"""
        print("正在初始化TF-IDF重排序器...")
        
        # 提取所有文档内容用于训练TF-IDF
        all_texts = []
        for doc in self.metadata:
            if 'metadata' in doc:
                content = doc['metadata'].get('original_content') or doc['metadata'].get('content', '')
            else:
                content = doc.get('original_content') or doc.get('content', '')
            all_texts.append(content)
        
        # 训练TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None,  # 保留中文停用词处理
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            max_df=0.95,
            min_df=2
        )
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            print(f"✓ TF-IDF重排序器初始化成功，特征维度: {self.tfidf_matrix.shape[1]}")
        except Exception as e:
            print(f"⚠️ TF-IDF重排序器初始化失败: {e}，将禁用重排序功能")
            self.enable_reranking = False
    
    def _expand_query(self, query: str) -> List[str]:
        """查询扩展：生成相关查询词"""
        if not self.enable_query_expansion:
            return [query]
        
        # 简单的查询扩展策略
        expanded_queries = [query]
        
        # 添加同义词和相关词（这里使用简单的规则，实际可以使用更复杂的方法）
        synonyms_map = {
            '公司': ['企业', '集团', '机构'],
            '业务': ['经营', '运营', '事业'],
            '产品': ['商品', '服务', '项目'],
            '发展': ['增长', '扩展', '进步'],
            '技术': ['科技', '工艺', '方法'],
            '市场': ['行业', '领域', '板块']
        }
        
        for word, synonyms in synonyms_map.items():
            if word in query:
                for synonym in synonyms:
                    expanded_query = query.replace(word, synonym)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # 限制扩展查询数量
    
    def _rerank_documents(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """使用TF-IDF重新排序检索到的文档"""
        if not self.enable_reranking or not hasattr(self, 'tfidf_vectorizer'):
            return retrieved_docs
        
        try:
            # 向量化查询
            query_tfidf = self.tfidf_vectorizer.transform([query])
            
            # 计算查询与每个候选文档的TF-IDF相似度
            rerank_scores = []
            for doc in retrieved_docs:
                doc_idx = doc['index']
                if doc_idx < self.tfidf_matrix.shape[0]:
                    doc_tfidf = self.tfidf_matrix[doc_idx:doc_idx+1]
                    tfidf_sim = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
                    
                    # 结合原始向量相似度和TF-IDF相似度
                    vector_sim = 1.0 / (1.0 + doc['distance'])  # 距离转换为相似度
                    combined_score = 0.7 * vector_sim + 0.3 * tfidf_sim
                    
                    rerank_scores.append((doc, combined_score))
                else:
                    rerank_scores.append((doc, doc.get('distance', float('inf'))))
            
            # 按组合分数重新排序
            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, score in rerank_scores]
            
            return reranked_docs[:self.final_k]
            
        except Exception as e:
            print(f"⚠️ 重排序失败: {e}，使用原始排序")
            return retrieved_docs[:self.final_k]
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型以选择合适的提示词模板"""
        query_lower = query.lower()
        
        # 事实性问题
        if any(word in query_lower for word in ['什么', '哪些', '多少', '何时', '何地', '谁']):
            return 'factual'
        
        # 分析性问题
        elif any(word in query_lower for word in ['为什么', '如何', '怎样', '分析', '原因', '影响']):
            return 'analytical'
        
        # 比较性问题
        elif any(word in query_lower for word in ['比较', '对比', '区别', '差异', '优劣']):
            return 'comparative'
        
        # 预测性问题
        elif any(word in query_lower for word in ['预测', '未来', '趋势', '前景', '展望']):
            return 'predictive'
        
        return 'factual'  # 默认为事实性问题
    
    def _build_adaptive_prompt(self, question: str, context: str, query_type: str) -> str:
        """根据问题类型构建自适应提示词"""
        
        base_instruction = "请严格根据以下'上下文信息'来回答'问题'。如果上下文中没有足够信息，请回答'根据提供的信息，无法回答该问题。'"
        
        type_specific_instructions = {
            'factual': "请提供准确的事实信息，包括具体的数据、时间、地点等细节。",
            'analytical': "请进行深入分析，解释原因、机制或影响因素，并提供逻辑推理过程。",
            'comparative': "请详细比较各项的异同点，突出关键差异和各自特点。",
            'predictive': "请基于现有信息分析趋势，但要明确指出这是基于当前信息的分析。"
        }
        
        specific_instruction = type_specific_instructions.get(query_type, type_specific_instructions['factual'])
        
        prompt_template = f"""{base_instruction}

{specific_instruction}

请注意：
1. 只能使用上下文中提供的信息来回答
2. 不要添加上下文中没有的信息
3. 如果信息不完整或不确定，请明确说明
4. 回答要条理清晰，重点突出

上下文信息：
{context}

问题：{question}

请根据上述要求回答："""
        
        return prompt_template
    
    def _build_dynamic_context(self, retrieved_docs: List[Dict], max_length: int = 4000) -> str:
        """动态构建上下文，优化长度和相关性"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            # 提取文档内容
            if 'metadata' in doc:
                metadata = doc['metadata']
                content = metadata.get('original_content') or metadata.get('content', '')
                filename = metadata.get('source_filename', '')
                page = metadata.get('source_page_number', 0)
            else:
                content = doc.get('original_content') or doc.get('content', '')
                filename = doc.get('source_filename', '')
                page = doc.get('source_page_number', 0)
            
            # 添加来源信息
            source_info = f"[文档{i+1}: {filename}, 第{page}页]"
            doc_with_source = f"{source_info}\n{content}"
            
            # 检查长度限制
            if current_length + len(doc_with_source) <= max_length:
                context_parts.append(doc_with_source)
                current_length += len(doc_with_source)
            else:
                # 如果超出长度限制，尝试截取部分内容
                remaining = max_length - current_length
                if remaining > 200:  # 至少保留200字符
                    truncated_content = content[:remaining-len(source_info)-50] + "..."
                    context_parts.append(f"{source_info}\n{truncated_content}")
                break
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """增强版问答方法"""
        if not question or not question.strip():
            return {
                "answer": "请提供有效的问题。",
                "filename": "",
                "page": 0,
                "sources": [],
                "performance_metrics": {}
            }
        
        start_time = time.time()
        print(f"处理问题: {question}")
        
        try:
            # 步骤1: 查询扩展
            retrieval_start = time.time()
            expanded_queries = self._expand_query(question)
            print(f"步骤1: 查询扩展完成，生成 {len(expanded_queries)} 个查询")
            
            # 步骤2: 多查询检索
            all_retrieved_docs = []
            for i, query in enumerate(expanded_queries):
                print(f"  正在处理查询 {i+1}: {query[:50]}...")
                
                # 向量化查询
                query_embeddings = self.embedding_client.embed([query])
                if not query_embeddings:
                    continue
                
                query_vector = np.array(query_embeddings[0], dtype=np.float32).reshape(1, -1)
                
                # 向量检索
                distances, indices = self.index.search(query_vector, self.k)
                
                # 收集检索结果
                for idx, distance in zip(indices[0], distances[0]):
                    if idx < len(self.metadata):
                        doc = self.metadata[idx].copy()
                        doc['index'] = idx
                        doc['distance'] = float(distance)
                        doc['query_source'] = i  # 记录来自哪个查询
                        all_retrieved_docs.append(doc)
            
            # 去重（基于文档索引）
            seen_indices = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                if doc['index'] not in seen_indices:
                    unique_docs.append(doc)
                    seen_indices.add(doc['index'])
            
            retrieval_time = time.time() - retrieval_start
            print(f"✓ 多查询检索完成，找到 {len(unique_docs)} 个唯一文档")
            
            # 步骤3: 重排序
            rerank_start = time.time()
            if self.enable_reranking:
                reranked_docs = self._rerank_documents(question, unique_docs)
                print(f"✓ 重排序完成，最终选择 {len(reranked_docs)} 个文档")
            else:
                reranked_docs = unique_docs[:self.final_k]
            
            rerank_time = time.time() - rerank_start
            
            if not reranked_docs:
                return {
                    "answer": "抱歉，没有找到相关的文档来回答您的问题。",
                    "filename": "",
                    "page": 0,
                    "sources": [],
                    "performance_metrics": {
                        "total_time": time.time() - start_time,
                        "retrieval_time": retrieval_time,
                        "rerank_time": rerank_time
                    }
                }
            
            # 步骤4: 动态上下文构建
            context_start = time.time()
            context = self._build_dynamic_context(reranked_docs)
            context_time = time.time() - context_start
            print(f"✓ 动态上下文构建完成，长度: {len(context)} 字符")
            
            # 步骤5: 自适应提示词构建
            query_type = self._classify_query_type(question)
            prompt = self._build_adaptive_prompt(question, context, query_type)
            print(f"✓ 自适应提示词构建完成，问题类型: {query_type}")
            
            # 步骤6: 生成答案
            generation_start = time.time()
            answer = self.generation_client.generate(prompt, temperature=0.1, max_tokens=1000)
            generation_time = time.time() - generation_start
            print(f"✓ 答案生成完成，长度: {len(answer)} 字符")
            
            # 步骤7: 格式化输出
            primary_source = reranked_docs[0] if reranked_docs else {}
            
            # 构建详细的来源信息
            sources_info = []
            for doc in reranked_docs:
                if 'metadata' in doc:
                    metadata = doc['metadata']
                    filename = metadata.get('source_filename', '')
                    page = metadata.get('source_page_number', 0)
                else:
                    filename = doc.get('source_filename', '')
                    page = doc.get('source_page_number', 0)
                
                sources_info.append({
                    'index': doc['index'],
                    'distance': doc['distance'],
                    'filename': filename,
                    'page': page,
                    'query_source': doc.get('query_source', 0)
                })
            
            total_time = time.time() - start_time
            
            # 性能指标
            performance_metrics = {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "context_time": context_time,
                "generation_time": generation_time,
                "expanded_queries_count": len(expanded_queries),
                "initial_docs_count": len(unique_docs),
                "final_docs_count": len(reranked_docs),
                "query_type": query_type
            }
            
            # 记录性能指标
            self.performance_metrics.append({
                'question': question,
                'metrics': performance_metrics,
                'timestamp': time.time()
            })
            
            result = {
                "answer": answer,
                "filename": primary_source.get('metadata', {}).get('source_filename', '') if 'metadata' in primary_source else primary_source.get('source_filename', ''),
                "page": primary_source.get('metadata', {}).get('source_page_number', 0) if 'metadata' in primary_source else primary_source.get('source_page_number', 0),
                "sources": sources_info,
                "performance_metrics": performance_metrics
            }
            
            print("✓ 增强版问答完成！\n")
            return result
            
        except Exception as e:
            print(f"✗ 问答过程中发生错误: {e}")
            return {
                "answer": f"抱歉，处理您的问题时发生错误: {str(e)}",
                "filename": "",
                "page": 0,
                "sources": [],
                "performance_metrics": {
                    "total_time": time.time() - start_time,
                    "error": str(e)
                }
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_metrics:
            return {"message": "暂无性能数据"}
        
        recent_metrics = self.performance_metrics[-50:]  # 最近50次查询
        
        total_times = [m['metrics']['total_time'] for m in recent_metrics]
        retrieval_times = [m['metrics']['retrieval_time'] for m in recent_metrics]
        generation_times = [m['metrics']['generation_time'] for m in recent_metrics]
        
        return {
            "total_queries": len(self.performance_metrics),
            "recent_queries": len(recent_metrics),
            "avg_total_time": np.mean(total_times),
            "avg_retrieval_time": np.mean(retrieval_times),
            "avg_generation_time": np.mean(generation_times),
            "optimization_features": {
                "query_expansion": self.enable_query_expansion,
                "reranking": self.enable_reranking,
                "initial_k": self.k,
                "final_k": self.final_k
            }
        }

# 使用示例
if __name__ == "__main__":
    try:
        # 初始化增强版RAG管道
        enhanced_rag = EnhancedRAGPipeline(
            k=10,  # 初始检索更多文档
            final_k=5,  # 最终返回5个
            enable_reranking=True,
            enable_query_expansion=True
        )
        
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
            print(f"性能指标: 总耗时 {result['performance_metrics']['total_time']:.2f}s")
            print(f"{'='*80}")
        
        # 显示性能报告
        print("\n" + "="*80)
        print("性能报告:")
        report = enhanced_rag.get_performance_report()
        for key, value in report.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"增强版RAG管道测试失败: {e}")