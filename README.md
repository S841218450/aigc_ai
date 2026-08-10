# AIGC Platform

基于 LangGraph + LangChain 的 AIGC 平台，支持文生图、文生视频、文生文以及知识库功能。

## 项目结构

```
aigc-ai/
├── app/
│   ├── api/                    # API 路由层
│   │   └── v1/                 # API v1 版本
│   │       └── endpoints/      # API 端点
│   ├── core/                   # 核心模块
│   │   ├── agents/             # 代理定义
│   │   ├── chains/             # 链定义
│   │   └── prompts/            # 提示词模板
│   ├── models/                 # 数据模型
│   │   ├── schemas/            # Pydantic 模型
│   │   └── entities/           # 数据库实体
│   ├── workflows/              # LangGraph 工作流
│   │   ├── text_to_image/      # 文生图工作流
│   │   ├── text_to_video/      # 文生视频工作流
│   │   ├── text_to_text/       # 文生文工作流
│   │   └── knowledge_base/     # 知识库工作流
│   ├── tools/                  # 工具集
│   │   ├── image_generation/   # 图像生成工具
│   │   ├── video_generation/   # 视频生成工具
│   │   ├── text_generation/    # 文本生成工具
│   │   └── retrieval/          # 检索工具
│   ├── utils/                  # 工具函数
│   ├── config/                 # 配置管理
│   └── main.py                 # FastAPI 应用入口
├── requirements.txt            # 依赖列表
├── .env.example                # 环境变量示例
└── pyproject.toml              # 项目配置
```

## 功能模块

### 1. 文生图 (Text to Image)
- 支持 Stable Diffusion 等图像生成模型
- 可配置生成参数（尺寸、风格等）

### 2. 文生视频 (Text to Video)
- 支持视频生成模型
- 可配置视频参数（时长、分辨率等）

### 3. 文生文 (Text to Text)
- 支持多种 LLM 模型
- 可配置生成参数（温度、最大长度等）

### 4. 知识库 (Knowledge Base)
- 支持文档检索和问答
- 使用向量数据库存储和检索文档

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量
```bash
cp .env .env
# 编辑 .env 文件，填入 API Key
```

3. 启动服务
```bash
uvicorn app.main:app --reload
```

4. 访问 API 文档
```
http://localhost:8000/docs
```

## API 接口

- `POST /api/v1/text-to-image/generate` - 文生图
- `POST /api/v1/text-to-video/generate` - 文生视频
- `POST /api/v1/text-to-text/generate` - 文生文
- `POST /api/v1/knowledge-base/query` - 知识库查询