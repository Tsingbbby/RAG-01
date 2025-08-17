#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分块与向量化主流程

该脚本负责读取Task 1的输出，执行分块策略，并调用向量化模块生成最终的vectorized_chunks.jsonl文件。
"""

import json
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入自定义模块
from embedding_module import EmbeddingModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkAndEmbedPipeline:
    """
    分块与向量化流水线
    
    负责处理PDF解析结果，执行分块策略，并生成向量化的知识块。
    """
    
    def __init__(self, 
                 input_file: str = "output/all_pdf_page_chunks.json",
                 output_file: str = "output/vectorized_chunks.jsonl",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 batch_size: int = 32):
        """
        初始化流水线
        
        Args:
            input_file: 输入的JSON文件路径
            output_file: 输出的JSONL文件路径
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            batch_size: 向量化批处理大小
        """
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ";", ";", ".", "!", "?", " ", ""]
        )
        
        # 初始化向量化模型
        logger.info("正在初始化向量化模型...")
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def load_input_data(self) -> Dict[str, Any]:
        """
        加载输入数据
        
        Returns:
            解析后的JSON数据
        """
        logger.info(f"正在加载输入文件: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载数据，包含 {len(data)} 个文档")
        return data
    
    def process_text_content(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理文本内容，执行分块策略
        
        Args:
            text: 原始文本内容
            metadata: 元数据信息
            
        Returns:
            分块后的数据列表
        """
        if not text or not text.strip():
            return []
        
        # 使用文本分割器进行分块
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for chunk in chunks:
            if chunk.strip():  # 跳过空块
                chunk_data = {
                    "content_for_embedding": chunk.strip(),
                    "metadata": {
                        **metadata,
                        "type": "text",
                        "original_content": chunk.strip()
                    }
                }
                result.append(chunk_data)
        
        return result
    
    def process_table_content(self, table_markdown: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理表格内容
        
        Args:
            table_markdown: 表格的Markdown表示
            metadata: 元数据信息
            
        Returns:
            表格块数据列表
        """
        if not table_markdown or not table_markdown.strip():
            return []
        
        chunk_data = {
            "content_for_embedding": table_markdown.strip(),
            "metadata": {
                **metadata,
                "type": "table",
                "original_content": table_markdown.strip()
            }
        }
        
        return [chunk_data]
    
    def process_image_content(self, image_caption: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理图片内容
        
        Args:
            image_caption: 图片的Caption描述
            metadata: 元数据信息
            
        Returns:
            图片块数据列表
        """
        if not image_caption or not image_caption.strip():
            return []
        
        chunk_data = {
            "content_for_embedding": image_caption.strip(),
            "metadata": {
                **metadata,
                "type": "image",
                "original_content": image_caption.strip()
            }
        }
        
        return [chunk_data]
    
    def extract_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从输入数据中提取所有块
        
        Args:
            data: 输入的JSON数据
            
        Returns:
            所有块的列表
        """
        chunks_to_embed = []
        
        logger.info("正在提取和处理内容块...")
        
        for filename, document in tqdm(data.items(), desc="处理文档"):
            if not isinstance(document, dict) or 'pages' not in document:
                logger.warning(f"跳过无效文档: {filename}")
                continue
            
            for page_num, page_content in document['pages'].items():
                page_number = int(page_num)
                
                # 基础元数据
                base_metadata = {
                    "source_filename": filename,
                    "source_page_number": page_number
                }
                
                # 处理文本内容
                if 'text' in page_content and page_content['text']:
                    text_chunks = self.process_text_content(
                        page_content['text'], 
                        base_metadata
                    )
                    chunks_to_embed.extend(text_chunks)
                
                # 处理表格内容
                if 'tables' in page_content:
                    for table in page_content['tables']:
                        if isinstance(table, dict) and 'markdown' in table:
                            table_chunks = self.process_table_content(
                                table['markdown'],
                                base_metadata
                            )
                            chunks_to_embed.extend(table_chunks)
                        elif isinstance(table, str):
                            # 如果表格直接是字符串格式
                            table_chunks = self.process_table_content(
                                table,
                                base_metadata
                            )
                            chunks_to_embed.extend(table_chunks)
                
                # 处理图片内容
                if 'images' in page_content:
                    for image in page_content['images']:
                        if isinstance(image, dict) and 'caption' in image:
                            image_chunks = self.process_image_content(
                                image['caption'],
                                base_metadata
                            )
                            chunks_to_embed.extend(image_chunks)
                        elif isinstance(image, str):
                            # 如果图片直接是字符串格式
                            image_chunks = self.process_image_content(
                                image,
                                base_metadata
                            )
                            chunks_to_embed.extend(image_chunks)
        
        logger.info(f"共提取到 {len(chunks_to_embed)} 个内容块")
        return chunks_to_embed
    
    def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对所有块进行向量化
        
        Args:
            chunks: 待向量化的块列表
            
        Returns:
            包含向量的完整块列表
        """
        if not chunks:
            return []
        
        logger.info("正在执行批量向量化...")
        
        # 提取所有待向量化的文本
        texts_to_embed = [chunk["content_for_embedding"] for chunk in chunks]
        
        # 批量向量化
        vectors = self.embedding_model.embed(texts_to_embed)
        
        # 将向量添加到块中并生成唯一ID
        vectorized_chunks = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            vectorized_chunk = {
                "chunk_id": str(uuid.uuid4()),
                "content_for_embedding": chunk["content_for_embedding"],
                "vector": vector,
                "metadata": chunk["metadata"]
            }
            vectorized_chunks.append(vectorized_chunk)
        
        return vectorized_chunks
    
    def save_results(self, vectorized_chunks: List[Dict[str, Any]]) -> None:
        """
        保存向量化结果到JSONL文件
        
        Args:
            vectorized_chunks: 向量化后的块列表
        """
        logger.info(f"正在保存结果到: {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for chunk in tqdm(vectorized_chunks, desc="保存块"):
                json_line = json.dumps(chunk, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"成功保存 {len(vectorized_chunks)} 个向量化块")
    
    def run(self) -> None:
        """
        执行完整的分块与向量化流程
        """
        try:
            # 1. 加载输入数据
            data = self.load_input_data()
            
            # 2. 提取和处理块
            chunks = self.extract_chunks(data)
            
            if not chunks:
                logger.warning("没有找到可处理的内容块")
                return
            
            # 3. 向量化
            vectorized_chunks = self.vectorize_chunks(chunks)
            
            # 4. 保存结果
            self.save_results(vectorized_chunks)
            
            # 5. 输出统计信息
            self._print_statistics(vectorized_chunks)
            
            logger.info("分块与向量化流程完成！")
            
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}")
            raise
    
    def _print_statistics(self, vectorized_chunks: List[Dict[str, Any]]) -> None:
        """
        打印统计信息
        
        Args:
            vectorized_chunks: 向量化后的块列表
        """
        if not vectorized_chunks:
            return
        
        # 统计不同类型的块数量
        type_counts = {}
        for chunk in vectorized_chunks:
            chunk_type = chunk["metadata"]["type"]
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # 统计向量维度
        vector_dim = len(vectorized_chunks[0]["vector"])
        
        logger.info("=== 处理统计 ===")
        logger.info(f"总块数: {len(vectorized_chunks)}")
        logger.info(f"向量维度: {vector_dim}")
        logger.info("块类型分布:")
        for chunk_type, count in type_counts.items():
            logger.info(f"  {chunk_type}: {count}")
        logger.info(f"输出文件: {self.output_file}")

def main():
    """
    主函数
    """
    # 检查输入文件是否存在
    input_file = "output/all_pdf_page_chunks.json"
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        logger.error("请先运行Task 1的PDF预处理流程生成该文件")
        return
    
    # 创建并运行流水线
    pipeline = ChunkAndEmbedPipeline(
        input_file=input_file,
        output_file="output/vectorized_chunks.jsonl",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=32
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()