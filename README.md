# 多模态RAG项目 - 环境配置与数据预处理

本项目实现了一个完整的多模态检索增强生成(RAG)系统的基础环境配置和PDF数据预处理流程。

## 项目结构

```
RAG-01/
├── datas/                          # 存放待处理的PDF文件
├── output/                         # 存放生成的数据文件
│   └── all_pdf_page_chunks.json   # PDF解析结果
├── scripts/                        # 存放可执行脚本
├── venv/                          # Python虚拟环境
├── environment_setup.ipynb        # 环境配置notebook
├── pdf_preprocessing.ipynb        # PDF预处理notebook
├── requirements.txt               # 项目依赖
├── .gitignore                     # Git忽略文件
└── README.md                      # 项目说明
```

## 快速开始

### 1. 环境配置

首先运行环境配置notebook：

```bash
# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 启动Jupyter
jupyter notebook environment_setup.ipynb
```

### 2. 数据预处理

将PDF文件放入 `datas/` 目录，然后运行预处理notebook：

```bash
jupyter notebook pdf_preprocessing.ipynb
```

## 功能特性

### 环境配置 (environment_setup.ipynb)

- ✅ 自动检测Python环境
- ✅ 安装核心依赖库（mineru, transformers, faiss-gpu等）
- ✅ 验证依赖安装状态
- ✅ 生成requirements.txt文件
- ✅ 显示项目目录结构

### PDF预处理 (pdf_preprocessing.ipynb)

- ✅ 批量处理PDF文件
- ✅ 使用MinerU进行文档解析
- ✅ 错误处理和重试机制
- ✅ 文本分块处理
- ✅ 生成结构化JSON输出
- ✅ 处理进度显示
- ✅ 详细的处理日志

## 核心依赖

- **mineru**: PDF文档解析
- **transformers**: 文本处理和模型支持
- **faiss-gpu**: 向量相似度搜索
- **tqdm**: 进度条显示
- **numpy**: 数值计算
- **PyPDF2**: 基础PDF处理（备用）

## 输出说明

### all_pdf_page_chunks.json 结构

```json
{
  "processing_info": {
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:05:00",
    "total_files": 5,
    "processed_files": 5,
    "failed_files": 0,
    "success_files": ["document1.pdf", "document2.pdf"],
    "total_pages": 50,
    "total_chunks": 200
  },
  "documents": [
    {
      "file_name": "document1.pdf",
      "file_path": "/path/to/document1.pdf",
      "file_size": 1024000,
      "parsed_at": "2024-01-01T10:01:00",
      "total_pages": 10,
      "total_chunks": 40,
      "pages": [
        {
          "page_number": 1,
          "text": "页面文本内容...",
          "text_length": 1500,
          "chunks": [
            {
              "chunk_id": "document1_page_1_chunk_1",
              "content": "文本块内容...",
              "start_pos": 0,
              "end_pos": 500
            }
          ]
        }
      ],
      "tables": [],
      "images": [],
      "metadata": {}
    }
  ]
}
```

## 使用说明

1. **准备PDF文件**: 将需要处理的PDF文件放入 `datas/` 目录
2. **运行环境配置**: 执行 `environment_setup.ipynb` 安装依赖
3. **执行预处理**: 运行 `pdf_preprocessing.ipynb` 处理PDF文件
4. **检查输出**: 在 `output/` 目录查看生成的 `all_pdf_page_chunks.json` 文件

## 错误处理

- 📝 **损坏的PDF**: 自动跳过并记录错误信息
- 🔄 **重试机制**: 对临时错误进行重试
- 📊 **详细日志**: 完整的处理过程记录
- ✅ **幂等性**: 可重复执行，覆盖旧结果

## 注意事项

- 确保在虚拟环境中运行所有代码
- PDF文件较大时处理时间可能较长
- 建议先用小文件测试流程
- 检查 `output/` 目录的磁盘空间

## 下一步

完成数据预处理后，可以继续进行：
- 文本向量化
- 构建向量索引
- 实现检索功能
- 集成生成模型

## 技术支持

如遇到问题，请检查：
1. Python版本兼容性
2. 依赖库安装状态
3. PDF文件格式和完整性
4. 磁盘空间和权限

---

**项目状态**: ✅ 环境配置与数据预处理完成
