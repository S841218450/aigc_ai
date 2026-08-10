# 项目规则 (Project Rules)

## 业务端接口调用

- 调用业务端（Java 后端）接口时，**必须**统一使用 [app/api/java_client.py](app/api/java_client.py) 中导出的 `API` 封装（`API.post / API.get / API.put`），禁止自行创建 `httpx`/`http.client` 等客户端实例、手动拼 URL 或重复写鉴权逻辑。
- 业务端响应为统一格式 `{code, msg, data, success}`，已由 `response_handler` 归一化；调用方用 `response.get("success")` 判断结果、`response["data"]` 取业务数据，不要再对原始响应做二次拆包。
