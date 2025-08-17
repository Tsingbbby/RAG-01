#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 测试脚本: 验证向量知识库构建功能

此脚本用于测试build_index.py的各项功能，包括:
1. 数据加载功能测试
2. FAISS索引构建测试
3. 文件保存和加载测试
4. 错误处理测试
"""

import json
import pickle
import numpy as np
import faiss
import os
import tempfile
import shutil
from typing import List, Dict, Any
from build_index import VectorKnowledgeBaseBuilder


class TestVectorKnowledgeBaseBuilder:
    """向量知识库构建器测试类"""
    
    def __init__(self):
        self.test_dir = None
        self.test_input_file = None
        
    def setup_test_environment(self) -> None:
        """设置测试环境"""
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp(prefix="test_build_index_")
        self.test_input_file = os.path.join(self.test_dir, "test_vectorized_chunks.jsonl")
        print(f"测试目录: {self.test_dir}")
    
    def cleanup_test_environment(self) -> None:
        """清理测试环境"""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"已清理测试目录: {self.test_dir}")
    
    def create_test_data(self, num_vectors: int = 10, vector_dim: int = 384) -> None:
        """创建测试数据"""
        print(f"创建测试数据: {num_vectors} 个向量，维度 {vector_dim}")
        
        test_data = []
        for i in range(num_vectors):
            # 生成随机向量
            vector = np.random.rand(vector_dim).astype(np.float32).tolist()
            
            # 创建测试数据项
            data_item = {
                "chunk_id": f"test_chunk_{i:03d}",
                "content": f"这是测试内容块 {i}",
                "vector": vector,
                "metadata": {
                    "source_filename": f"test_doc_{i // 3}.pdf",
                    "page_number": i % 5 + 1,
                    "chunk_index": i,
                    "content_type": "text"
                }
            }
            test_data.append(data_item)
        
        # 写入JSONL文件
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"测试数据已保存到: {self.test_input_file}")
    
    def test_data_loading(self) -> None:
        """测试数据加载功能"""
        print("\n=== 测试数据加载功能 ===")
        
        builder = VectorKnowledgeBaseBuilder(
            input_file=self.test_input_file,
            output_dir=self.test_dir
        )
        
        try:
            vector_array, metadata_list = builder.load_vectorized_data()
            
            # 验证数据
            assert vector_array.shape[0] == 10, f"向量数量错误: {vector_array.shape[0]}"
            assert vector_array.shape[1] == 384, f"向量维度错误: {vector_array.shape[1]}"
            assert len(metadata_list) == 10, f"元数据数量错误: {len(metadata_list)}"
            assert vector_array.dtype == np.float32, f"向量类型错误: {vector_array.dtype}"
            
            # 验证元数据结构
            for i, metadata in enumerate(metadata_list):
                assert 'chunk_id' in metadata, "元数据缺少chunk_id"
                assert 'metadata' in metadata, "元数据缺少metadata字段"
                assert metadata['chunk_id'] == f"test_chunk_{i:03d}", "chunk_id不匹配"
            
            print("✅ 数据加载测试通过")
            return vector_array, metadata_list
            
        except Exception as e:
            print(f"❌ 数据加载测试失败: {e}")
            raise
    
    def test_faiss_index_building(self, vector_array: np.ndarray) -> faiss.Index:
        """测试FAISS索引构建"""
        print("\n=== 测试FAISS索引构建 ===")
        
        builder = VectorKnowledgeBaseBuilder(
            input_file=self.test_input_file,
            output_dir=self.test_dir
        )
        
        try:
            index = builder.build_faiss_index(vector_array)
            
            # 验证索引
            assert index.ntotal == 10, f"索引向量数量错误: {index.ntotal}"
            assert index.d == 384, f"索引维度错误: {index.d}"
            
            # 测试搜索功能
            query_vector = vector_array[0:1]  # 使用第一个向量作为查询
            distances, indices = index.search(query_vector, k=3)
            
            assert len(distances[0]) == 3, "搜索结果数量错误"
            assert indices[0][0] == 0, "最相似向量应该是自己"
            assert distances[0][0] < 1e-6, "自相似距离应该接近0"
            
            print("✅ FAISS索引构建测试通过")
            return index
            
        except Exception as e:
            print(f"❌ FAISS索引构建测试失败: {e}")
            raise
    
    def test_file_saving_and_loading(self, index: faiss.Index, metadata_list: List[Dict[str, Any]]) -> None:
        """测试文件保存和加载"""
        print("\n=== 测试文件保存和加载 ===")
        
        builder = VectorKnowledgeBaseBuilder(
            input_file=self.test_input_file,
            output_dir=self.test_dir
        )
        
        try:
            # 保存文件
            builder.save_outputs(index, metadata_list)
            
            # 验证文件存在
            assert os.path.exists(builder.index_file), "索引文件未创建"
            assert os.path.exists(builder.metadata_file), "元数据文件未创建"
            
            # 测试加载索引
            loaded_index = faiss.read_index(builder.index_file)
            assert loaded_index.ntotal == index.ntotal, "加载的索引向量数量不匹配"
            assert loaded_index.d == index.d, "加载的索引维度不匹配"
            
            # 测试加载元数据
            with open(builder.metadata_file, 'rb') as f:
                loaded_metadata = pickle.load(f)
            
            assert len(loaded_metadata) == len(metadata_list), "加载的元数据数量不匹配"
            
            # 验证元数据内容
            for i, (original, loaded) in enumerate(zip(metadata_list, loaded_metadata)):
                assert original['chunk_id'] == loaded['chunk_id'], f"第{i}个chunk_id不匹配"
                assert original['metadata'] == loaded['metadata'], f"第{i}个metadata不匹配"
            
            print("✅ 文件保存和加载测试通过")
            
        except Exception as e:
            print(f"❌ 文件保存和加载测试失败: {e}")
            raise
    
    def test_error_handling(self) -> None:
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        # 测试文件不存在的情况
        try:
            builder = VectorKnowledgeBaseBuilder(
                input_file="nonexistent_file.jsonl",
                output_dir=self.test_dir
            )
            builder.load_vectorized_data()
            assert False, "应该抛出FileNotFoundError"
        except FileNotFoundError:
            print("✅ 文件不存在错误处理正确")
        except Exception as e:
            print(f"❌ 文件不存在错误处理失败: {e}")
            raise
        
        # 测试空文件的情况
        try:
            empty_file = os.path.join(self.test_dir, "empty.jsonl")
            with open(empty_file, 'w') as f:
                pass  # 创建空文件
            
            builder = VectorKnowledgeBaseBuilder(
                input_file=empty_file,
                output_dir=self.test_dir
            )
            builder.load_vectorized_data()
            assert False, "应该抛出ValueError"
        except ValueError as e:
            if "输入文件为空" in str(e):
                print("✅ 空文件错误处理正确")
            else:
                raise
        except Exception as e:
            print(f"❌ 空文件错误处理失败: {e}")
            raise
    
    def test_complete_workflow(self) -> None:
        """测试完整工作流程"""
        print("\n=== 测试完整工作流程 ===")
        
        builder = VectorKnowledgeBaseBuilder(
            input_file=self.test_input_file,
            output_dir=self.test_dir
        )
        
        try:
            # 执行完整构建流程
            builder.build()
            
            # 验证输出文件
            assert os.path.exists(builder.index_file), "索引文件未创建"
            assert os.path.exists(builder.metadata_file), "元数据文件未创建"
            
            # 验证可以正常加载和使用
            index = faiss.read_index(builder.index_file)
            with open(builder.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # 测试搜索功能
            query_vector = np.random.rand(1, 384).astype(np.float32)
            distances, indices = index.search(query_vector, k=5)
            
            assert len(distances[0]) == 5, "搜索结果数量错误"
            assert all(0 <= idx < len(metadata) for idx in indices[0]), "索引超出范围"
            
            print("✅ 完整工作流程测试通过")
            
        except Exception as e:
            print(f"❌ 完整工作流程测试失败: {e}")
            raise
    
    def run_all_tests(self) -> None:
        """运行所有测试"""
        try:
            print("开始向量知识库构建功能测试...")
            
            # 设置测试环境
            self.setup_test_environment()
            
            # 创建测试数据
            self.create_test_data()
            
            # 运行各项测试
            vector_array, metadata_list = self.test_data_loading()
            index = self.test_faiss_index_building(vector_array)
            self.test_file_saving_and_loading(index, metadata_list)
            self.test_error_handling()
            self.test_complete_workflow()
            
            print("\n🎉 所有测试通过！向量知识库构建功能正常")
            
        except Exception as e:
            print(f"\n💥 测试失败: {e}")
            raise
        finally:
            # 清理测试环境
            self.cleanup_test_environment()


def main():
    """主函数"""
    tester = TestVectorKnowledgeBaseBuilder()
    tester.run_all_tests()


if __name__ == "__main__":
    main()