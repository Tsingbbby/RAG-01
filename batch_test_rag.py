#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试RAG管道脚本 - Task 6 升级版
支持基线和增强模式的A/B对比测试
"""

import json
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from rag_pipeline import RAGPipeline
from enhanced_rag_pipeline import EnhancedRAGPipeline
import numpy as np

# 加载环境变量
load_dotenv()

def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def json_serializer(obj):
    """JSON序列化转换函数，处理numpy数据类型"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def save_results(results, output_file):
    """保存测试结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_serializer)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RAG管道批量测试工具')
    parser.add_argument('--mode', choices=['baseline', 'enhanced'], default='baseline',
                       help='选择测试模式: baseline (基线RAG) 或 enhanced (增强RAG)')
    parser.add_argument('--test-file', default='test.json',
                       help='测试数据文件路径 (默认: test.json)')
    parser.add_argument('--output-dir', default='output',
                       help='输出目录 (默认: output)')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Task 6: RAG管道批量测试 - {args.mode.upper()} 模式")
    print("=" * 60)
    
    # 检查必要文件
    test_file = args.test_file
    if not os.path.exists(test_file):
        print(f"错误：找不到测试数据文件 {test_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载测试数据
    print(f"正在加载测试数据: {test_file}")
    test_data = load_test_data(test_file)
    print(f"✓ 加载完成，共 {len(test_data)} 个测试样本")
    
    # 初始化性能日志
    log_entries = []
    start_time = datetime.now()
    log_entries.append(f"批量测试开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append(f"测试模式: {args.mode.upper()}")
    log_entries.append(f"测试数据集: {test_file}")
    log_entries.append(f"总问题数: {len(test_data)}")
    
    # 初始化RAG管道
    print(f"\n正在初始化{args.mode.upper()}模式RAG管道...")
    try:
        if args.mode == 'baseline':
            rag = RAGPipeline(
                index_path="output/knowledge_base.index",
                metadata_path="output/chunk_metadata.pkl",
                k=5
            )
            log_entries.append("管道类型: RAGPipeline (基线版)")
            log_entries.append("配置: k=5")
        else:  # enhanced mode
            rag = EnhancedRAGPipeline(
                index_path="output/knowledge_base.index",
                metadata_path="output/chunk_metadata.pkl",
                initial_k=20,
                final_k=5,
                enable_reranking=True
            )
            log_entries.append("管道类型: EnhancedRAGPipeline (增强版)")
            log_entries.append("配置: initial_k=20, final_k=5, reranking=True")
        
        print(f"✓ {args.mode.upper()}模式RAG管道初始化成功")
    except Exception as e:
        print(f"✗ RAG管道初始化失败: {e}")
        return
    
    # 批量测试
    print(f"\n开始批量测试 {len(test_data)} 个问题...")
    results = []  # 详细测试结果
    submission_results = []  # submission.json格式结果
    
    successful_count = 0
    failed_count = 0
    total_processing_time = 0
    
    # 使用tqdm显示进度条
    for i, item in enumerate(tqdm(test_data, desc="处理问题"), 1):
        question_start_time = time.time()
        
        try:
            # 使用RAG管道回答问题
            result = rag.answer_question(item['question'])
            question_end_time = time.time()
            processing_time = question_end_time - question_start_time
            total_processing_time += processing_time
            
            # 记录详细结果
            test_result = {
                "id": i,
                "filename": item['filename'],
                "page": item['page'],
                "question": item['question'],
                "expected_answer": item['answer'],
                "rag_answer": result['answer'],
                "sources": result['sources'],
                "processing_time": processing_time,
                "success": True
            }
            
            # 记录submission格式结果
            submission_result = {
                "answer": result['answer'],
                "filename": item['filename'],
                "page": item['page']
            }
            
            successful_count += 1
            log_entries.append(f"问题 {i}: 成功 (耗时: {processing_time:.2f}s)")
            
        except Exception as e:
            question_end_time = time.time()
            processing_time = question_end_time - question_start_time
            total_processing_time += processing_time
            
            error_msg = f"处理失败: {str(e)}"
            
            # 记录详细结果
            test_result = {
                "id": i,
                "filename": item['filename'],
                "page": item['page'],
                "question": item['question'],
                "expected_answer": item['answer'],
                "rag_answer": error_msg,
                "sources": [],
                "processing_time": processing_time,
                "success": False
            }
            
            # 记录submission格式结果（失败时使用默认值）
            submission_result = {
                "answer": "根据提供的信息，无法回答该问题。",
                "filename": item['filename'],
                "page": item['page']
            }
            
            failed_count += 1
            log_entries.append(f"问题 {i}: 失败 - {str(e)} (耗时: {processing_time:.2f}s)")
        
        results.append(test_result)
        submission_results.append(submission_result)
    
    # 计算总体性能指标
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    avg_processing_time = total_processing_time / len(test_data) if test_data else 0
    success_rate = (successful_count / len(test_data) * 100) if test_data else 0
    
    # 添加性能统计到日志
    log_entries.extend([
        f"批量测试结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_duration:.2f}秒",
        f"成功回答数: {successful_count}",
        f"失败数: {failed_count}",
        f"成功率: {success_rate:.1f}%",
        f"平均处理时间: {avg_processing_time:.2f}秒/问题",
        f"总处理时间: {total_processing_time:.2f}秒"
    ])
    
    # 保存submission.json文件
    submission_file = f"{args.output_dir}/submission_{args.mode}.json"
    print(f"\n正在保存提交文件: {submission_file}")
    with open(submission_file, 'w', encoding='utf-8') as f:
        json.dump(submission_results, f, ensure_ascii=False, indent=2, default=json_serializer)
    print(f"✓ 提交文件保存完成")
    
    # 保存详细测试结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_results_file = f"{args.output_dir}/rag_test_results_{args.mode}_{timestamp}.json"
    print(f"\n正在保存详细测试结果: {detailed_results_file}")
    save_results(results, detailed_results_file)
    print(f"✓ 详细结果保存完成")
    
    # 保存性能日志
    log_file = f"{args.output_dir}/{args.mode}_performance_log.txt"
    print(f"\n正在保存性能日志: {log_file}")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_entries))
    print(f"✓ 性能日志保存完成")
    
    # 显示统计结果
    print("\n" + "=" * 60)
    print(f"Task 6 批量测试完成统计 - {args.mode.upper()} 模式")
    print("=" * 60)
    print(f"测试模式: {args.mode.upper()}")
    print(f"总测试数量: {len(results)}")
    print(f"成功数量: {successful_count}")
    print(f"失败数量: {failed_count}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总耗时: {total_duration:.2f}秒")
    print(f"平均处理时间: {avg_processing_time:.2f}秒/问题")
    
    # 显示增强模式特有的性能指标
    if args.mode == 'enhanced' and hasattr(rag, 'get_performance_report'):
        print("\n增强模式性能报告:")
        report = rag.get_performance_report()
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n生成文件:")
    print(f"  - 提交文件: {submission_file}")
    print(f"  - 详细结果: {detailed_results_file}")
    print(f"  - 性能日志: {log_file}")
    print("=" * 60)
    
    # 提示用户如何进行对比测试
    if args.mode == 'baseline':
        print("\n💡 提示: 运行以下命令进行增强模式测试以便对比:")
        print(f"   python {os.path.basename(__file__)} --mode enhanced")
    elif args.mode == 'enhanced':
        print("\n💡 提示: 如需对比，请确保已运行基线模式测试:")
        print(f"   python {os.path.basename(__file__)} --mode baseline")
    print("=" * 60)

if __name__ == "__main__":
    main()