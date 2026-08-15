# 项目规则 (Project Rules)

## 业务端接口调用

- 调用业务端（Java 后端）接口时，**必须**统一使用 [app/api/java_client.py](app/api/java_client.py) 中导出的 `API` 封装（`API.post / API.get / API.put`），禁止自行创建 `httpx`/`http.client` 等客户端实例、手动拼 URL 或重复写鉴权逻辑。
- 业务端响应为统一格式 `{code, msg, data, success}`，已由 `response_handler` 归一化；调用方用 `response.get("success")` 判断结果、`response["data"]` 取业务数据，不要再对原始响应做二次拆包。

## 工作流分层约定

设计/新增工作流时，代码**必须**按以下层次严格放置，禁止在单文件里混装：

- 工作流接线（节点/路由组装、`StateGraph` 构建）→ `app/workflows/<业务>/graph.py`
- 路由选择（条件分支函数，如 `conditional edges` 的 router）→ `app/workflows/<业务>/router.py`
- 节点实现 → `app/workflows/<业务>/nodes.py`
- 工具调用 → `app/workflows/<业务>/tools/`，**根据用途区分不同文件**（如 `doc_tools.py` 文档检索、`table_tools.py` 表查询）
- 公共节点/工具（多个工作流复用的）→ `app/workflows/common/`（如 `common_node.py` agent 工厂、`agent_stream.py` 流式辅助）
