#!/usr/bin/env python3
"""
基线测试脚本 - 评估基线RAGPipeline在验证集上的性能
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tasks.task4_rag_pipeline.rag_pipeline import RAGPipeline

def load_validation_set(validation_file):
    """加载验证集"""
    with open(validation_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_baseline_results(results, output_file):
    """保存基线测试结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def evaluate_baseline():
    """评估基线性能"""
    print("开始基线性能评估...")
    
    # 文件路径
    validation_file = "tasks/task6_quality_sprint/validation_set_20.json"
    output_file = "tasks/task6_quality_sprint/baseline_results.json"
    
    # 加载验证集
    validation_data = load_validation_set(validation_file)
    print(f"加载了 {len(validation_data)} 个验证问题")
    
    # 初始化基线RAG管道
    print("初始化基线RAG管道...")
    rag_pipeline = RAGPipeline()
    
    # 测试结果
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": "baseline",
            "validation_set_size": len(validation_data)
        },
        "results": []
    }
    
    # 逐个处理验证问题
    for i, item in enumerate(validation_data, 1):
        print(f"\n处理问题 {i}/{len(validation_data)}: {item['question'][:50]}...")
        
        try:
            # 获取基线答案
            start_time = datetime.now()
            baseline_answer = rag_pipeline.answer_question(item['question'])
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # 记录结果
            result = {
                "question_id": i,
                "filename": item['filename'],
                "page": item['page'],
                "question": item['question'],
                "standard_answer": item['answer'],
                "baseline_answer": baseline_answer,
                "processing_time_seconds": processing_time,
                "status": "success"
            }
            
            print(f"✓ 处理完成，耗时: {processing_time:.2f}秒")
            
        except Exception as e:
            print(f"✗ 处理失败: {str(e)}")
            result = {
                "question_id": i,
                "filename": item['filename'],
                "page": item['page'],
                "question": item['question'],
                "standard_answer": item['answer'],
                "baseline_answer": f"ERROR: {str(e)}",
                "processing_time_seconds": 0,
                "status": "error"
            }
        
        results["results"].append(result)
    
    # 计算统计信息
    successful_results = [r for r in results["results"] if r["status"] == "success"]
    if successful_results:
        avg_time = sum(r["processing_time_seconds"] for r in successful_results) / len(successful_results)
        total_time = sum(r["processing_time_seconds"] for r in successful_results)
        
        results["statistics"] = {
            "total_questions": len(validation_data),
            "successful_questions": len(successful_results),
            "failed_questions": len(validation_data) - len(successful_results),
            "success_rate": len(successful_results) / len(validation_data),
            "average_processing_time_seconds": avg_time,
            "total_processing_time_seconds": total_time
        }
    
    # 保存结果
    save_baseline_results(results, output_file)
    print(f"\n基线测试完成！结果已保存到: {output_file}")
    
    # 打印统计信息
    if "statistics" in results:
        stats = results["statistics"]
        print(f"\n=== 基线性能统计 ===")
        print(f"总问题数: {stats['total_questions']}")
        print(f"成功处理: {stats['successful_questions']}")
        print(f"失败数量: {stats['failed_questions']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"平均处理时间: {stats['average_processing_time_seconds']:.2f}秒")
        print(f"总处理时间: {stats['total_processing_time_seconds']:.2f}秒")

if __name__ == "__main__":
    evaluate_baseline()