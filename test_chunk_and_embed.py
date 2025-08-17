#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分块与向量化流程

该脚本用于测试chunk_and_embed_pipeline.py的功能，
包括创建测试数据和验证输出格式。
"""

import json
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """
    创建测试用的all_pdf_page_chunks.json文件
    """
    test_data = {
        "test_document.pdf": {
            "pages": {
                "1": {
                    "text": "这是第一页的文本内容。它包含了一些基本的信息，用于测试文本分块功能。文本应该被分割成适当大小的块，每个块大约500个字符，并且有50个字符的重叠。这样可以确保信息的连续性和完整性。",
                    "tables": [
                        {
                            "markdown": "| 列1 | 列2 | 列3 |\n|-----|-----|-----|\n| 数据1 | 数据2 | 数据3 |\n| 数据4 | 数据5 | 数据6 |"
                        }
                    ],
                    "images": [
                        {
                            "caption": "这是一个示例图片的描述，展示了某个重要的概念或数据可视化结果。"
                        }
                    ]
                },
                "2": {
                    "text": "第二页包含更多的文本内容。这些内容将被用来测试分块算法的效果。我们需要确保每个文本块都有合适的大小，既不会太长也不会太短。同时，重叠部分应该能够保持上下文的连贯性，这对于后续的检索任务非常重要。",
                    "tables": [],
                    "images": []
                }
            }
        },
        "another_document.pdf": {
            "pages": {
                "1": {
                    "text": "另一个文档的内容，用于测试多文档处理能力。",
                    "tables": [
                        "| 产品 | 价格 | 库存 |\n|------|------|------|\n| 产品A | 100元 | 50 |\n| 产品B | 200元 | 30 |"
                    ],
                    "images": [
                        "产品展示图：显示了各种产品的外观和特征。"
                    ]
                }
            }
        }
    }
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 保存测试数据
    with open("output/all_pdf_page_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info("测试数据已创建: output/all_pdf_page_chunks.json")

def validate_output():
    """
    验证输出文件的格式和内容
    """
    output_file = "output/vectorized_chunks.jsonl"
    
    if not os.path.exists(output_file):
        logger.error(f"输出文件不存在: {output_file}")
        return False
    
    logger.info(f"正在验证输出文件: {output_file}")
    
    chunk_count = 0
    required_fields = ["chunk_id", "content_for_embedding", "vector", "metadata"]
    required_metadata_fields = ["source_filename", "source_page_number", "type", "original_content"]
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    chunk = json.loads(line)
                    chunk_count += 1
                    
                    # 验证必需字段
                    for field in required_fields:
                        if field not in chunk:
                            logger.error(f"第{line_num}行缺少字段: {field}")
                            return False
                    
                    # 验证元数据字段
                    metadata = chunk.get("metadata", {})
                    for field in required_metadata_fields:
                        if field not in metadata:
                            logger.error(f"第{line_num}行元数据缺少字段: {field}")
                            return False
                    
                    # 验证数据类型
                    if not isinstance(chunk["chunk_id"], str):
                        logger.error(f"第{line_num}行chunk_id应为字符串")
                        return False
                    
                    if not isinstance(chunk["content_for_embedding"], str):
                        logger.error(f"第{line_num}行content_for_embedding应为字符串")
                        return False
                    
                    if not isinstance(chunk["vector"], list):
                        logger.error(f"第{line_num}行vector应为列表")
                        return False
                    
                    if not all(isinstance(x, (int, float)) for x in chunk["vector"]):
                        logger.error(f"第{line_num}行vector应包含数值")
                        return False
                    
                    if metadata["type"] not in ["text", "table", "image"]:
                        logger.error(f"第{line_num}行类型应为text、table或image")
                        return False
                    
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行JSON格式错误: {e}")
                    return False
        
        logger.info(f"验证通过！共处理 {chunk_count} 个块")
        
        # 统计信息
        type_counts = {}
        vector_dims = set()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk_type = chunk["metadata"]["type"]
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                    vector_dims.add(len(chunk["vector"]))
        
        logger.info("=== 输出统计 ===")
        logger.info(f"总块数: {chunk_count}")
        logger.info(f"向量维度: {list(vector_dims)}")
        logger.info("类型分布:")
        for chunk_type, count in type_counts.items():
            logger.info(f"  {chunk_type}: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"验证过程中出错: {e}")
        return False

def main():
    """
    主测试函数
    """
    logger.info("开始测试分块与向量化流程")
    
    # 1. 创建测试数据
    create_test_data()
    
    # 2. 运行主流程
    logger.info("运行分块与向量化流程...")
    try:
        from chunk_and_embed_pipeline import ChunkAndEmbedPipeline
        
        pipeline = ChunkAndEmbedPipeline(
            input_file="output/all_pdf_page_chunks.json",
            output_file="output/vectorized_chunks.jsonl",
            chunk_size=200,  # 使用较小的块大小进行测试
            chunk_overlap=20,
            batch_size=4
        )
        
        pipeline.run()
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        return
    
    # 3. 验证输出
    if validate_output():
        logger.info("测试通过！")
    else:
        logger.error("测试失败！")

if __name__ == "__main__":
    main()