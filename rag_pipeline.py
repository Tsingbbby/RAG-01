import os
import pickle
import numpy as np
from typing import Dict, Any, List
import faiss
from dotenv import load_dotenv

# 导入自定义模块
from embedding_module import EmbeddingAPIClient
from llm_api_client import GenerationAPIClient

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) 推理管道
    整合向量检索和文本生成，提供端到端的问答服务
    """
    
    def __init__(self, 
                 index_path: str = "output/knowledge_base.index",
                 metadata_path: str = "output/chunk_metadata.pkl",
                 k: int = 5):
        """
        初始化RAG管道，加载所有必要的资源
        
        Args:
            index_path: FAISS索引文件路径
            metadata_path: 元数据文件路径
            k: 检索返回的文档数量
        """
        self.k = k
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # 加载环境变量
        load_dotenv()
        
        print("正在初始化RAG管道...")
        
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
        
        print("RAG管道初始化完成！\n")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        回答用户问题的核心方法
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案、来源文件名和页码的字典
        """
        if not question or not question.strip():
            return {
                "answer": "请提供有效的问题。",
                "filename": "",
                "page": 0
            }
        
        print(f"处理问题: {question}")
        
        try:
            # 步骤1: 查询向量化
            print("步骤1: 正在向量化查询...")
            query_embeddings = self.embedding_client.embed([question])
            if not query_embeddings:
                raise ValueError("查询向量化失败")
            query_vector = np.array(query_embeddings[0], dtype=np.float32).reshape(1, -1)
            print(f"✓ 查询向量化完成，维度: {query_vector.shape[1]}")
            
            # 步骤2: 向量检索
            print(f"步骤2: 正在检索Top-{self.k}相关文档...")
            distances, indices = self.index.search(query_vector, self.k)
            print(f"✓ 检索完成，找到 {len(indices[0])} 个相关文档")
            
            # 步骤3: 构建上下文
            print("步骤3: 正在构建上下文...")
            contexts = []
            retrieved_docs = []
            
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.metadata):
                    doc = self.metadata[idx]
                    contexts.append(doc['original_content'])
                    retrieved_docs.append({
                        'index': idx,
                        'distance': float(distance),
                        'filename': doc.get('source_filename', ''),
                        'page': doc.get('source_page_number', 0),
                        'content_preview': doc['original_content'][:100] + '...' if len(doc['original_content']) > 100 else doc['original_content']
                    })
                    print(f"  文档{i+1}: {doc.get('source_filename', 'Unknown')} (页码: {doc.get('source_page_number', 'N/A')}, 距离: {distance:.4f})")
            
            if not contexts:
                return {
                    "answer": "抱歉，没有找到相关的文档来回答您的问题。",
                    "filename": "",
                    "page": 0
                }
            
            context = "\n\n".join(contexts)
            print(f"✓ 上下文构建完成，总长度: {len(context)} 字符")
            
            # 步骤4: 提示词工程
            print("步骤4: 正在构建提示词...")
            prompt = self._build_prompt(question, context)
            print(f"✓ 提示词构建完成，长度: {len(prompt)} 字符")
            
            # 步骤5: 生成答案
            print("步骤5: 正在生成答案...")
            answer = self.generation_client.generate(prompt, temperature=0.1, max_tokens=1000)
            print(f"✓ 答案生成完成，长度: {len(answer)} 字符")
            
            # 步骤6: 格式化输出
            print("步骤6: 正在格式化输出...")
            # 选择最相关的文档作为来源（Top-1）
            primary_source = retrieved_docs[0] if retrieved_docs else {}
            
            result = {
                "answer": answer,
                "filename": primary_source.get('filename', ''),
                "page": primary_source.get('page', 0)
            }
            
            print("✓ 问答完成！\n")
            return result
            
        except Exception as e:
            print(f"✗ 问答过程中发生错误: {e}")
            return {
                "answer": f"抱歉，处理您的问题时发生错误: {str(e)}",
                "filename": "",
                "page": 0
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建结构化的提示词模板
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            
        Returns:
            格式化的提示词
        """
        prompt_template = """请严格根据以下"上下文信息"来回答"问题"。如果上下文中没有足够信息，请回答"根据提供的信息，无法回答该问题。"

请注意：
1. 只能使用上下文中提供的信息来回答
2. 不要添加上下文中没有的信息
3. 如果信息不完整或不确定，请明确说明
4. 回答要准确、简洁、有条理

---
上下文信息:
{context}
---

问题:
{question}

回答:"""
        
        return prompt_template.format(context=context, question=question)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        获取管道配置信息
        
        Returns:
            管道配置信息字典
        """
        return {
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "k": self.k,
            "index_dimension": self.index.d if hasattr(self, 'index') else None,
            "total_documents": len(self.metadata) if hasattr(self, 'metadata') else None,
            "embedding_model": self.embedding_client.model_name if hasattr(self, 'embedding_client') else None,
            "generation_model": self.generation_client.model_name if hasattr(self, 'generation_client') else None
        }
    
    def test_pipeline(self) -> bool:
        """
        测试管道是否正常工作
        
        Returns:
            测试是否成功
        """
        try:
            test_question = "测试问题"
            result = self.answer_question(test_question)
            return 'answer' in result and len(result['answer']) > 0
        except Exception as e:
            print(f"管道测试失败: {e}")
            return False

# 测试代码
if __name__ == "__main__":
    try:
        # 初始化RAG管道
        rag = RAGPipeline()
        
        # 显示管道信息
        info = rag.get_pipeline_info()
        print("管道配置信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # 测试问答
        test_questions = [
            "公司在2023年的重大产品临床进展是什么？",
            "公司的主要业务是什么？",
            "公司有哪些重要的研发项目？"
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            result = rag.answer_question(question)
            print(f"\n问题: {question}")
            print(f"答案: {result['answer']}")
            print(f"来源: {result['filename']} (第{result['page']}页)")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"RAG管道测试失败: {e}")