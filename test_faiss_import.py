#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试faiss模块导入和基本功能
用于验证云端环境中的faiss-cpu安装是否正常
"""

import sys
import os

def test_faiss_import():
    """测试faiss模块导入"""
    try:
        import faiss
        print(f"✅ faiss导入成功")
        print(f"📦 faiss版本: {faiss.__version__}")
        
        # 测试基本功能
        import numpy as np
        
        # 创建一个简单的向量索引
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        print(f"✅ 创建索引成功，维度: {dimension}")
        
        # 添加一些测试向量
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        print(f"✅ 添加向量成功，索引中向量数量: {index.ntotal}")
        
        # 测试搜索
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=3)
        print(f"✅ 搜索成功，返回距离: {distances[0][:3]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ faiss导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ faiss功能测试失败: {e}")
        return False

def test_project_modules():
    """测试项目模块导入"""
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print("\n=== 测试项目模块导入 ===")
        
        # 测试embedding_module
        from embedding_module import EmbeddingAPIClient
        print("✅ embedding_module导入成功")
        
        # 测试llm_api_client
        from llm_api_client import GenerationAPIClient
        print("✅ llm_api_client导入成功")
        
        # 测试rag_pipeline
        from rag_pipeline import RAGPipeline
        print("✅ rag_pipeline导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 项目模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 项目模块测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== Python环境信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    print("\n=== 测试faiss模块 ===")
    faiss_ok = test_faiss_import()
    
    if faiss_ok:
        project_ok = test_project_modules()
        
        if project_ok:
            print("\n🎉 所有模块导入测试通过！")
            print("💡 现在可以在Jupyter Notebook中正常使用RAG pipeline了")
        else:
            print("\n⚠️  faiss模块正常，但项目模块导入有问题")
    else:
        print("\n❌ faiss模块导入失败，请检查安装")

if __name__ == "__main__":
    main()