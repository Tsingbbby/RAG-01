#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG管道测试脚本
测试检索与生成的核心推理逻辑
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline
from llm_api_client import GenerationAPIClient
from embedding_module import EmbeddingAPIClient

def test_api_clients():
    """
    测试API客户端连接
    """
    print("\n" + "="*60)
    print("测试 API 客户端连接")
    print("="*60)
    
    # 测试Embedding API
    try:
        embedding_client = EmbeddingAPIClient()
        print(f"✓ Embedding API客户端初始化成功")
        print(f"  模型: {embedding_client.model_name}")
        print(f"  端点: {embedding_client.endpoint_url}")
        
        # 测试embedding
        test_texts = ["这是一个测试文本"]
        embeddings = embedding_client.embed(test_texts)
        print(f"✓ Embedding测试成功，向量维度: {len(embeddings[0])}")
        
    except Exception as e:
        print(f"✗ Embedding API测试失败: {e}")
        return False
    
    # 测试Generation API
    try:
        generation_client = GenerationAPIClient()
        print(f"✓ Generation API客户端初始化成功")
        print(f"  模型: {generation_client.model_name}")
        print(f"  端点: {generation_client.endpoint_url}")
        
        # 测试生成
        if generation_client.test_connection():
            print(f"✓ Generation API连接测试成功")
        else:
            print(f"✗ Generation API连接测试失败")
            return False
            
    except Exception as e:
        print(f"✗ Generation API测试失败: {e}")
        return False
    
    return True

def test_rag_pipeline_initialization():
    """
    测试RAG管道初始化
    """
    print("\n" + "="*60)
    print("测试 RAG 管道初始化")
    print("="*60)
    
    try:
        rag = RAGPipeline()
        
        # 显示管道信息
        info = rag.get_pipeline_info()
        print("\n管道配置信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return rag
        
    except Exception as e:
        print(f"✗ RAG管道初始化失败: {e}")
        return None

def test_question_answering(rag: RAGPipeline):
    """
    测试问答功能
    """
    print("\n" + "="*60)
    print("测试 问答功能")
    print("="*60)
    
    # 测试问题列表
    test_questions = [
        "公司在2023年的重大产品临床进展是什么？",
        "公司的主要业务领域有哪些？",
        "公司有哪些重要的研发项目？",
        "公司的财务状况如何？",
        "公司的核心竞争优势是什么？"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = rag.answer_question(question)
            end_time = time.time()
            
            print(f"答案: {result['answer']}")
            print(f"来源: {result['filename']} (第{result['page']}页)")
            print(f"耗时: {end_time - start_time:.2f}秒")
            
            # 记录结果
            results.append({
                "question": question,
                "answer": result['answer'],
                "filename": result['filename'],
                "page": result['page'],
                "response_time": end_time - start_time,
                "success": True
            })
            
        except Exception as e:
            end_time = time.time()
            print(f"✗ 问答失败: {e}")
            print(f"耗时: {end_time - start_time:.2f}秒")
            
            results.append({
                "question": question,
                "error": str(e),
                "response_time": end_time - start_time,
                "success": False
            })
    
    return results

def test_edge_cases(rag: RAGPipeline):
    """
    测试边界情况
    """
    print("\n" + "="*60)
    print("测试 边界情况")
    print("="*60)
    
    edge_cases = [
        ("", "空字符串"),
        ("   ", "空白字符串"),
        ("这是一个完全不相关的问题关于火星探索", "不相关问题"),
        ("a" * 1000, "超长问题"),
        ("什么？", "极短问题")
    ]
    
    for question, description in edge_cases:
        print(f"\n测试 {description}: '{question[:50]}{'...' if len(question) > 50 else ''}'")
        
        try:
            result = rag.answer_question(question)
            print(f"✓ 处理成功: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
        except Exception as e:
            print(f"✗ 处理失败: {e}")

def save_test_results(results: List[Dict[str, Any]]):
    """
    保存测试结果
    """
    output_file = "test_results_rag_pipeline.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 测试结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n✗ 保存测试结果失败: {e}")

def generate_test_report(results: List[Dict[str, Any]]):
    """
    生成测试报告
    """
    print("\n" + "="*60)
    print("测试报告")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    avg_response_time = sum(r['response_time'] for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"总测试数: {total_tests}")
    print(f"成功: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"失败: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print(f"平均响应时间: {avg_response_time:.2f}秒")
    
    if failed_tests > 0:
        print("\n失败的测试:")
        for i, result in enumerate(results, 1):
            if not result.get('success', False):
                print(f"  {i}. {result['question'][:50]}... - {result.get('error', 'Unknown error')}")

def main():
    """
    主测试函数
    """
    print("RAG管道测试开始")
    print("="*60)
    
    # 检查必要文件
    required_files = [
        "output/knowledge_base.index",
        "output/chunk_metadata.pkl",
        ".env"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"✗ 缺少必要文件: {missing_files}")
        print("请确保已完成Task 2和Task 3，并配置了.env文件")
        return
    
    # 测试API客户端
    if not test_api_clients():
        print("\n✗ API客户端测试失败，终止测试")
        return
    
    # 测试RAG管道初始化
    rag = test_rag_pipeline_initialization()
    if rag is None:
        print("\n✗ RAG管道初始化失败，终止测试")
        return
    
    # 测试问答功能
    results = test_question_answering(rag)
    
    # 测试边界情况
    test_edge_cases(rag)
    
    # 生成测试报告
    generate_test_report(results)
    
    # 保存测试结果
    save_test_results(results)
    
    print("\n" + "="*60)
    print("RAG管道测试完成")
    print("="*60)

if __name__ == "__main__":
    main()