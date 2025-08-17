#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3: 构建与索引向量知识库

此脚本负责将Task 2生成的向量化知识块构建成可进行高效相似度搜索的向量知识库。
主要功能:
1. 读取vectorized_chunks.jsonl文件
2. 构建FAISS索引
3. 生成knowledge_base.index和chunk_metadata.pkl文件
"""

import json
import pickle
import numpy as np
import faiss
import os
from typing import List, Dict, Any


class VectorKnowledgeBaseBuilder:
    """向量知识库构建器"""
    
    def __init__(self, input_file: str = "output/vectorized_chunks.jsonl", 
                 output_dir: str = "output"):
        """
        初始化构建器
        
        Args:
            input_file: 输入的向量化数据文件路径
            output_dir: 输出目录
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.index_file = os.path.join(output_dir, "knowledge_base.index")
        self.metadata_file = os.path.join(output_dir, "chunk_metadata.pkl")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def load_vectorized_data(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        加载向量化数据
        
        Returns:
            tuple: (向量数组, 元数据列表)
        """
        print(f"正在加载向量化数据: {self.input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"输入文件不存在: {self.input_file}\n"
                f"请先运行Task 2生成向量化数据文件。"
            )
        
        vectors_list = []
        metadata_list = []
        
        # 逐行读取JSONL文件
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON对象
                    data = json.loads(line)
                    
                    # 验证必需字段
                    if 'vector' not in data:
                        raise ValueError(f"第{line_num}行缺少'vector'字段")
                    if 'chunk_id' not in data:
                        raise ValueError(f"第{line_num}行缺少'chunk_id'字段")
                    if 'metadata' not in data:
                        raise ValueError(f"第{line_num}行缺少'metadata'字段")
                    
                    # 提取向量和元数据
                    vector = data['vector']
                    chunk_id = data['chunk_id']
                    metadata = data['metadata']
                    
                    # 验证向量格式
                    if not isinstance(vector, list) or not vector:
                        raise ValueError(f"第{line_num}行的向量格式无效")
                    
                    # 添加到列表
                    vectors_list.append(vector)
                    metadata_list.append({
                        'chunk_id': chunk_id,
                        'metadata': metadata
                    })
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"第{line_num}行JSON解析错误: {e}")
                except Exception as e:
                    raise ValueError(f"第{line_num}行处理错误: {e}")
        
        if not vectors_list:
            raise ValueError("输入文件为空或没有有效数据")
        
        # 转换为NumPy数组
        print(f"成功加载 {len(vectors_list)} 个向量")
        vector_array = np.array(vectors_list, dtype=np.float32)
        
        # 验证向量维度一致性
        if len(vector_array.shape) != 2:
            raise ValueError("向量维度不一致")
        
        print(f"向量维度: {vector_array.shape}")
        return vector_array, metadata_list
    
    def build_faiss_index(self, vector_array: np.ndarray) -> faiss.Index:
        """
        构建FAISS索引
        
        Args:
            vector_array: 向量数组
            
        Returns:
            构建好的FAISS索引
        """
        print("正在构建FAISS索引...")
        
        # 获取向量维度
        d = vector_array.shape[1]
        print(f"向量维度: {d}")
        
        # 初始化FAISS索引 (使用L2距离)
        index = faiss.IndexFlatL2(d)
        print(f"创建IndexFlatL2索引，维度: {d}")
        
        # 检查GPU可用性
        gpu_count = faiss.get_num_gpus()
        print(f"检测到 {gpu_count} 个GPU")
        
        if gpu_count > 0:
            print("使用GPU加速索引构建...")
            # 转换为GPU索引
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            
            # 添加向量到GPU索引
            gpu_index.add(vector_array)
            
            # 转回CPU索引以便保存
            index = faiss.index_gpu_to_cpu(gpu_index)
            print("GPU索引构建完成，已转回CPU")
        else:
            print("使用CPU构建索引...")
            # 直接在CPU上添加向量
            index.add(vector_array)
            print("CPU索引构建完成")
        
        print(f"索引中的向量数量: {index.ntotal}")
        return index
    
    def save_outputs(self, index: faiss.Index, metadata_list: List[Dict[str, Any]]) -> None:
        """
        保存输出文件
        
        Args:
            index: FAISS索引
            metadata_list: 元数据列表
        """
        print("正在保存输出文件...")
        
        # 保存FAISS索引
        print(f"保存索引文件: {self.index_file}")
        faiss.write_index(index, self.index_file)
        
        # 保存元数据映射
        print(f"保存元数据文件: {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata_list, f)
        
        print("所有文件保存完成")
        
        # 输出文件信息
        index_size = os.path.getsize(self.index_file) / (1024 * 1024)  # MB
        metadata_size = os.path.getsize(self.metadata_file) / (1024 * 1024)  # MB
        
        print(f"\n=== 输出文件信息 ===")
        print(f"索引文件: {self.index_file} ({index_size:.2f} MB)")
        print(f"元数据文件: {self.metadata_file} ({metadata_size:.2f} MB)")
        print(f"向量数量: {index.ntotal}")
        print(f"向量维度: {index.d}")
    
    def build(self) -> None:
        """
        执行完整的构建流程
        """
        try:
            print("=== Task 3: 构建向量知识库 ===")
            
            # 1. 加载数据
            vector_array, metadata_list = self.load_vectorized_data()
            
            # 2. 验证数据对齐
            if len(vector_array) != len(metadata_list):
                raise ValueError(
                    f"数据对齐错误: 向量数量({len(vector_array)}) != "
                    f"元数据数量({len(metadata_list)})"
                )
            
            # 3. 构建FAISS索引
            index = self.build_faiss_index(vector_array)
            
            # 4. 保存输出文件
            self.save_outputs(index, metadata_list)
            
            print("\n✅ 向量知识库构建成功!")
            
        except Exception as e:
            print(f"\n❌ 构建失败: {e}")
            raise


def main():
    """主函数"""
    builder = VectorKnowledgeBaseBuilder()
    builder.build()


if __name__ == "__main__":
    main()