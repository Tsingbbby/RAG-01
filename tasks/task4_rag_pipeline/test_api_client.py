#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API客户端功能
验证EmbeddingAPIClient与硅基流动/Xinference的兼容性
"""

import os
import sys
from embedding_module import EmbeddingAPIClient

def test_api_client():
    """
    测试API客户端的基本功能
    """
    print("=== 测试EmbeddingAPIClient ===")
    
    try:
        # 初始化客户端
        print("1. 初始化API客户端...")
        client = EmbeddingAPIClient()
        print(f"   - API端点: {client.endpoint_url}")
        print(f"   - 模型名称: {client.model_name}")
        print(f"   - API密钥: {'已设置' if client.api_key else '未设置'}")
        
        # 测试单个文本向量化
        print("\n2. 测试单个文本向量化...")
        test_text = ["这是一个测试文本"]
        embedding = client.embed(test_text)
        print(f"   - 输入文本: {test_text[0]}")
        print(f"   - 向量维度: {len(embedding[0])}")
        print(f"   - 向量前5个值: {embedding[0][:5]}")
        
        # 测试批量文本向量化
        print("\n3. 测试批量文本向量化...")
        test_texts = [
            "这是第一个测试文本",
            "This is the second test text",
            "第三个测试文本包含中英文mixed content",
            "数据分块与多模态向量化",
            "API客户端兼容性测试"
        ]
        embeddings = client.embed(test_texts)
        print(f"   - 输入文本数量: {len(test_texts)}")
        print(f"   - 输出向量数量: {len(embeddings)}")
        print(f"   - 向量维度: {len(embeddings[0])}")
        
        # 验证向量一致性
        print("\n4. 验证向量一致性...")
        for i, text in enumerate(test_texts):
            if len(embeddings[i]) != len(embeddings[0]):
                print(f"   ❌ 向量维度不一致: 文本{i+1}")
                return False
        print("   ✅ 所有向量维度一致")
        
        # 测试获取向量维度
        print("\n5. 测试获取向量维度...")
        dimension = client.get_embedding_dimension()
        print(f"   - 向量维度: {dimension}")
        
        # 测试空输入处理
        print("\n6. 测试空输入处理...")
        empty_result = client.embed([])
        print(f"   - 空输入结果: {empty_result}")
        
        print("\n=== 测试完成 ✅ ===")
        print("API客户端功能正常，与远程服务兼容")
        return True
        
    except Exception as e:
        print(f"\n=== 测试失败 ❌ ===")
        print(f"错误信息: {e}")
        print("\n可能的解决方案:")
        print("1. 检查.env文件配置是否正确")
        print("2. 确认API服务是否可访问")
        print("3. 验证API密钥是否有效")
        print("4. 检查网络连接")
        return False

def check_environment():
    """
    检查环境配置
    """
    print("=== 环境配置检查 ===")
    
    # 检查.env文件
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ .env文件存在: {env_file}")
    else:
        print(f"❌ .env文件不存在: {env_file}")
        return False
    
    # 检查环境变量
    required_vars = ['LOCAL_API_KEY', 'LOCAL_BASE_URL', 'LOCAL_EMBEDDING_MODEL']
    from dotenv import load_dotenv
    load_dotenv()
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == 'LOCAL_API_KEY':
                print(f"✅ {var}: {'*' * min(len(value), 10)}...")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未设置")
            return False
    
    return True

if __name__ == "__main__":
    print("开始测试API客户端...\n")
    
    # 检查环境
    if not check_environment():
        print("\n环境配置检查失败，请先配置.env文件")
        sys.exit(1)
    
    print()
    
    # 运行测试
    success = test_api_client()
    
    if success:
        print("\n🎉 所有测试通过！API客户端已准备就绪")
        sys.exit(0)
    else:
        print("\n💥 测试失败，请检查配置和网络连接")
        sys.exit(1)