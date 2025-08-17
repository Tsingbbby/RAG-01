# RAG系统检索准确率优化指南

## 当前系统分析

基于对现有RAG管道的分析，当前系统采用了标准的检索-生成架构：
- 使用FAISS进行向量检索
- 基于余弦相似度的Top-K检索
- 简单的上下文拼接策略
- 固定的提示词模板

## 技术优化方案

### 1. 向量检索优化

#### 1.1 混合检索策略
```python
# 实现BM25 + 向量检索的混合方案
class HybridRetriever:
    def __init__(self, vector_index, bm25_index, alpha=0.7):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.alpha = alpha  # 向量检索权重
    
    def retrieve(self, query, k=5):
        # 向量检索
        vector_scores = self.vector_index.search(query, k*2)
        # BM25检索
        bm25_scores = self.bm25_index.search(query, k*2)
        # 分数融合
        combined_scores = self._combine_scores(vector_scores, bm25_scores)
        return combined_scores[:k]
```

#### 1.2 查询扩展
```python
def expand_query(self, query: str) -> List[str]:
    """使用同义词和相关词扩展查询"""
    # 方法1: 使用LLM生成相关查询
    expanded_queries = self.llm.generate_related_queries(query)
    
    # 方法2: 使用词向量找相似词
    similar_terms = self.word_embeddings.find_similar(query)
    
    return [query] + expanded_queries + similar_terms
```

#### 1.3 重排序机制
```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 5):
        """使用交叉编码器对候选文档重新排序"""
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in ranked_indices]
```

### 2. 文档分块优化

#### 2.1 语义分块
```python
class SemanticChunker:
    def __init__(self, embedding_model, similarity_threshold=0.8):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
    
    def chunk_by_semantics(self, text: str) -> List[str]:
        """基于语义相似度进行分块"""
        sentences = self.split_sentences(text)
        embeddings = self.embedding_model.embed(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i-1], embeddings[i]
            )
            
            if similarity > self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        chunks.append(' '.join(current_chunk))
        return chunks
```

#### 2.2 重叠窗口策略
```python
def create_overlapping_chunks(text: str, chunk_size: int = 512, overlap: int = 128):
    """创建带重叠的文档块"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks
```

### 3. 上下文优化

#### 3.1 动态上下文选择
```python
class DynamicContextBuilder:
    def __init__(self, max_context_length=4000):
        self.max_length = max_context_length
    
    def build_context(self, query: str, retrieved_docs: List[Dict]) -> str:
        """动态选择最相关的上下文片段"""
        # 计算每个文档与查询的相关性
        relevance_scores = self._calculate_relevance(query, retrieved_docs)
        
        # 按相关性排序
        sorted_docs = sorted(
            zip(retrieved_docs, relevance_scores),
            key=lambda x: x[1], reverse=True
        )
        
        # 动态选择上下文，确保不超过长度限制
        context_parts = []
        current_length = 0
        
        for doc, score in sorted_docs:
            doc_text = doc['content']
            if current_length + len(doc_text) <= self.max_length:
                context_parts.append(doc_text)
                current_length += len(doc_text)
            else:
                # 截取部分内容
                remaining = self.max_length - current_length
                if remaining > 100:  # 至少保留100字符
                    context_parts.append(doc_text[:remaining])
                break
        
        return '\n\n'.join(context_parts)
```

#### 3.2 上下文压缩
```python
class ContextCompressor:
    def __init__(self, compression_model):
        self.model = compression_model
    
    def compress_context(self, query: str, context: str) -> str:
        """压缩上下文，保留与查询最相关的信息"""
        prompt = f"""
        请从以下上下文中提取与问题最相关的关键信息：
        
        问题：{query}
        
        上下文：{context}
        
        请只保留能够回答问题的核心信息，去除无关内容：
        """
        
        compressed = self.model.generate(prompt, max_tokens=1000)
        return compressed
```

### 4. 提示词工程优化

#### 4.1 自适应提示词
```python
class AdaptivePromptBuilder:
    def __init__(self):
        self.templates = {
            'factual': self._factual_template,
            'analytical': self._analytical_template,
            'comparative': self._comparative_template
        }
    
    def build_prompt(self, query: str, context: str) -> str:
        """根据问题类型选择合适的提示词模板"""
        query_type = self._classify_query_type(query)
        template_func = self.templates.get(query_type, self._default_template)
        return template_func(query, context)
    
    def _classify_query_type(self, query: str) -> str:
        """分类问题类型"""
        if any(word in query.lower() for word in ['什么', '哪些', '多少']):
            return 'factual'
        elif any(word in query.lower() for word in ['为什么', '如何', '分析']):
            return 'analytical'
        elif any(word in query.lower() for word in ['比较', '对比', '区别']):
            return 'comparative'
        return 'factual'
```

#### 4.2 思维链提示
```python
def build_cot_prompt(self, query: str, context: str) -> str:
    """构建思维链提示词"""
    return f"""
    请按照以下步骤回答问题：
    
    1. 首先，仔细阅读提供的上下文信息
    2. 识别与问题相关的关键信息
    3. 逐步分析这些信息如何回答问题
    4. 给出最终答案
    
    上下文信息：
    {context}
    
    问题：{query}
    
    请按照上述步骤思考并回答：
    """
```

### 5. 评估与反馈优化

#### 5.1 检索质量评估
```python
class RetrievalEvaluator:
    def __init__(self):
        self.metrics = ['precision', 'recall', 'mrr', 'ndcg']
    
    def evaluate_retrieval(self, queries: List[str], 
                          retrieved_docs: List[List[str]], 
                          ground_truth: List[List[str]]) -> Dict[str, float]:
        """评估检索质量"""
        results = {}
        
        for metric in self.metrics:
            scores = []
            for q, ret, gt in zip(queries, retrieved_docs, ground_truth):
                score = self._calculate_metric(metric, ret, gt)
                scores.append(score)
            results[metric] = np.mean(scores)
        
        return results
```

#### 5.2 在线学习机制
```python
class OnlineLearner:
    def __init__(self, feedback_threshold=0.1):
        self.feedback_data = []
        self.threshold = feedback_threshold
    
    def collect_feedback(self, query: str, retrieved_docs: List[str], 
                        user_rating: float):
        """收集用户反馈"""
        self.feedback_data.append({
            'query': query,
            'docs': retrieved_docs,
            'rating': user_rating,
            'timestamp': time.time()
        })
    
    def update_retrieval_weights(self):
        """基于反馈更新检索权重"""
        if len(self.feedback_data) < 100:  # 需要足够的反馈数据
            return
        
        # 分析反馈模式，调整检索参数
        positive_feedback = [f for f in self.feedback_data if f['rating'] > 0.7]
        negative_feedback = [f for f in self.feedback_data if f['rating'] < 0.3]
        
        # 基于反馈调整检索策略
        self._adjust_retrieval_strategy(positive_feedback, negative_feedback)
```

### 6. 实施建议

#### 6.1 渐进式优化
1. **第一阶段**：实施重排序机制和查询扩展
2. **第二阶段**：优化文档分块策略
3. **第三阶段**：引入混合检索和动态上下文
4. **第四阶段**：实施在线学习和自适应优化

#### 6.2 性能监控
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = []
    
    def log_performance(self, query: str, retrieval_time: float, 
                       generation_time: float, relevance_score: float):
        """记录性能指标"""
        self.metrics_history.append({
            'query': query,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'relevance_score': relevance_score,
            'timestamp': time.time()
        })
    
    def get_performance_report(self) -> Dict[str, float]:
        """生成性能报告"""
        if not self.metrics_history:
            return {}
        
        recent_data = self.metrics_history[-100:]  # 最近100次查询
        
        return {
            'avg_retrieval_time': np.mean([d['retrieval_time'] for d in recent_data]),
            'avg_generation_time': np.mean([d['generation_time'] for d in recent_data]),
            'avg_relevance_score': np.mean([d['relevance_score'] for d in recent_data]),
            'total_queries': len(self.metrics_history)
        }
```

## 预期效果

通过实施上述优化方案，预期可以实现：

1. **检索准确率提升20-30%**：通过混合检索和重排序
2. **上下文相关性提升25%**：通过动态上下文选择
3. **回答质量提升15-20%**：通过自适应提示词工程
4. **系统鲁棒性增强**：通过在线学习和反馈机制

## 实施优先级

**高优先级**（立即实施）：
- 重排序机制
- 查询扩展
- 动态上下文选择

**中优先级**（1-2周内）：
- 语义分块
- 自适应提示词
- 性能监控

**低优先级**（长期规划）：
- 在线学习机制
- 混合检索架构
- 上下文压缩

---

*注：具体实施时需要根据实际数据和业务需求进行调整和测试。*