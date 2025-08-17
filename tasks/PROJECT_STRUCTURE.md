# RAG-01 项目文件组织结构

本文档说明了RAG系统项目按照任务（Task）顺序重新组织后的文件结构。

## 📁 目录结构概览

```
RAG-01/
├── tasks/                          # 按任务分类的文件夹
│   ├── task1_environment_setup/     # Task 1: 环境设置
│   ├── task2_pdf_processing/        # Task 2: PDF处理
│   ├── task3_build_index/          # Task 3: 索引构建
│   ├── task4_rag_pipeline/         # Task 4: RAG管道
│   └── task5_batch_testing/        # Task 5: 批量测试
├── datas/                          # 原始数据文件
├── scripts/                        # 通用脚本
├── output/                         # 输出结果（原始）
├── .env                           # 环境变量配置
├── .gitignore                     # Git忽略文件
├── README.md                      # 项目说明
└── PROJECT_STRUCTURE.md           # 本文档
```

## 📋 各Task详细内容

### Task 1: 环境设置 (`tasks/task1_environment_setup/`)

**目标**: 建立开发环境，安装依赖包，配置基础设施

**文件列表**:
- `environment_setup.ipynb` - 环境设置演示Notebook
- `requirements.txt` - Python依赖包列表

**关键功能**:
- Python环境配置
- 依赖包安装和版本管理
- 基础工具验证

### Task 2: PDF处理 (`tasks/task2_pdf_processing/`)

**目标**: 实现PDF文档的解析、预处理和文本提取

**文件列表**:
- `pdf_preprocessing.ipynb` - PDF预处理演示Notebook
- `generate_sample_data.py` - 样本数据生成脚本

**关键功能**:
- PDF文档解析
- 文本提取和清理
- 样本数据生成

### Task 3: 索引构建 (`tasks/task3_build_index/`)

**目标**: 构建向量索引，实现文档的向量化和检索基础设施

**文件列表**:
- `build_index.py` - 索引构建主脚本
- `chunk_and_embed_pipeline.py` - 文档分块和嵌入管道
- `embedding_module.py` - 嵌入模型封装
- `task3_build_index_demo.ipynb` - 索引构建演示Notebook
- `test_build_index.py` - 索引构建测试
- `test_chunk_and_embed.py` - 分块嵌入测试
- `test_faiss_import.py` - FAISS库导入测试

**关键功能**:
- 文档分块策略
- 向量嵌入生成
- FAISS索引构建
- 向量检索优化

### Task 4: RAG管道 (`tasks/task4_rag_pipeline/`)

**目标**: 实现完整的RAG（检索增强生成）管道

**文件列表**:
- `rag_pipeline.py` - 主RAG管道实现
- `rag_pipeline_no_faiss.py` - 无FAISS版本的RAG管道
- `enhanced_rag_pipeline.py` - 增强版RAG管道
- `llm_api_client.py` - LLM API客户端
- `rag_demo_faiss.py` - FAISS版RAG演示
- `task4_enhanced_rag_demo.ipynb` - 增强RAG演示Notebook
- `task4_rag_demo_no_faiss.ipynb` - 无FAISS RAG演示Notebook
- `task4_rag_pipeline_demo.ipynb` - RAG管道演示Notebook
- `test_rag_pipeline.py` - RAG管道测试
- `test_api_client.py` - API客户端测试
- `rag_optimization_guide.md` - RAG优化指南

**关键功能**:
- 查询理解和向量化
- 相关文档检索
- 上下文构建
- LLM答案生成
- 结果后处理

### Task 5: 批量测试 (`tasks/task5_batch_testing/`)

**目标**: 实现端到端的批量测试和性能评估

**文件列表**:
- `batch_test_rag.py` - 完整批量测试脚本
- `batch_test_rag_sample.py` - 样例批量测试脚本
- `task5_batch_testing_demo.ipynb` - 批量测试演示Notebook
- `test.json` - 测试数据集
- `train.json` - 训练数据集
- `output/` - 测试结果输出目录
  - `submission.json` - 最终提交文件
  - `submission_sample.json` - 样例提交文件
  - `rag_test_results_*.json` - 详细测试结果
  - `sample_performance_log.txt` - 性能日志

**关键功能**:
- 批量问答测试
- 性能监控和日志
- 结果格式化和验证
- 可视化分析和报告

## 🔄 文件组织原则

1. **任务导向**: 按照开发流程的自然顺序组织文件
2. **功能内聚**: 相关功能的文件放在同一目录
3. **易于导航**: 清晰的命名和层次结构
4. **版本控制友好**: 保持原有的Git历史

## 🚀 使用指南

### 开发流程
1. 从Task 1开始，按顺序完成各个任务
2. 每个Task目录包含该阶段的所有相关文件
3. 使用Notebook文件进行交互式开发和演示
4. 运行测试文件验证功能正确性

### 文件引用
- 在跨Task引用文件时，使用相对路径
- 核心模块（如`embedding_module.py`）可能被多个Task使用
- 配置文件（如`.env`）在项目根目录

### 输出管理
- 每个Task的输出保存在对应目录
- Task 5的`output/`目录包含最终测试结果
- 原始`output/`目录保留作为备份

## 📊 项目进度追踪

- ✅ Task 1: 环境设置 - 已完成
- ✅ Task 2: PDF处理 - 已完成  
- ✅ Task 3: 索引构建 - 已完成
- ✅ Task 4: RAG管道 - 已完成
- ✅ Task 5: 批量测试 - 已完成

## 🔧 维护说明

- 添加新功能时，将文件放入对应的Task目录
- 更新文档时，同步更新本结构说明
- 定期检查文件组织的合理性，必要时重构

---

**创建时间**: 2025-08-17  
**最后更新**: 2025-08-17  
**维护者**: RAG-01 开发团队