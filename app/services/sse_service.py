import asyncio
import json
import time
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from fastapi.responses import StreamingResponse

from app.services.node_status import get_step_name


class SSEEvent:
    """SSE 事件封装（带 seq_id 序列号）"""

    def __init__(
        self,
        event_type: str,
        status: str,
        data: Any = None,
        seq_id: int = 0,
        thread_id: Any = None,
    ):
        self.type = event_type
        self.status = status
        self.data = data
        self.seq_id = seq_id
        self.thread_id = thread_id
        self.timestamp = int(time.time() * 1000)

    def to_sse(self) -> str:
        payload = {
            "seq_id": self.seq_id,
            "type": self.type,
            "status": self.status,
            "data": self.data,
            "threadId": self.thread_id,
            "timestamp": self.timestamp,
        }
        body = json.dumps(payload, ensure_ascii=False)
        if len(body.encode("utf-8")) > 4096:
            payload["data"] = "[payload too large]"
            body = json.dumps(payload, ensure_ascii=False)
        # 标准 SSE 格式：id: 用 seq_id，event: 用 type，data: 用 JSON
        return f"id: {self.seq_id}\nevent: {self.type}\ndata: {body}\n\n"


class SSEService:
    """
    SSE 流式服务
    - 统一事件格式 + seq_id 自增序列号
    - 心跳保活（默认 15s）
    - 取消支持
    - 事件持久化（可选，用于断点续传）
    """

    def __init__(self, thread_id: Any = None, heartbeat_interval: int = 15, event_store=None):
        self.heartbeat_interval = heartbeat_interval
        self.thread_id = thread_id
        self._cancelled = False
        self._seq_counter = 0
        self._event_store = event_store  # 可选的事件存储服务

    def cancel(self):
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def last_seq_id(self) -> int:
        return self._seq_counter

    def next_event(self, event_type: str, status: str, data: Any = None) -> SSEEvent:
        self._seq_counter += 1
        return SSEEvent(
            event_type=event_type,
            status=status,
            data=data,
            seq_id=self._seq_counter,
            thread_id=self.thread_id,
        )

    async def stream(
        self,
        event_generator: AsyncGenerator[SSEEvent, None],
    ) -> AsyncGenerator[str, None]:
        """核心流式输出：将事件生成器转为 SSE 文本流，自动注入心跳"""
        pending_tasks = []  # 跟踪 fire-and-forget 的持久化任务
        try:
            agen = event_generator.__aiter__()
            pending = None

            while True:
                #是否取消
                if self._cancelled:
                    yield self.next_event("cancelled", "用户已取消").to_sse()
                    break

                if pending is None:
                    pending = asyncio.create_task(self._anext_safe(agen))

                heartbeat = asyncio.create_task(asyncio.sleep(self.heartbeat_interval))

                done, _ = await asyncio.wait(
                    {pending, heartbeat},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if heartbeat in done:
                    yield ": heartbeat\n\n"
                    continue

                event = pending.result()
                pending = None

                if event is None:
                    break

                # fire-and-forget 持久化：不阻塞 SSE 输出
                if self._event_store:
                    pending_tasks.append(asyncio.create_task(self._event_store.save(event)))

                yield event.to_sse()

        except asyncio.CancelledError:
            yield self.next_event("cancelled", "连接已断开").to_sse()
        except Exception:
            # 不透出内部错误原因给前端，仅提示服务异常
            yield self.next_event("error", "服务异常，请稍后重试").to_sse()
        finally:
            if pending and not pending.done():
                pending.cancel()
            # 等待所有持久化任务完成
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

    @staticmethod
    async def _anext_safe(agen):
        try:
            return await agen.__anext__()
        except StopAsyncIteration:
            return None


# ---- 便捷工厂方法（自动分配 seq_id） ----

def sse_event(sse: SSEService, event_type: str, status: str, data: Any = None) -> SSEEvent:
    return sse.next_event(event_type=event_type, status=status, data=data)


def sse_done(sse: SSEService, data: Any = None) -> SSEEvent:
    return sse.next_event(event_type="done", status="执行完成", data=data)


def sse_interrupt(sse: SSEService, thread_id: Any, interrupt_data: Any) -> SSEEvent:
    return sse.next_event(
        event_type="human_in_the_loop",
        status="需要人工介入",
        data={
            "threadId": thread_id,
            "interrupt": interrupt_data,
        },
    )


# ---- 通用工具：断点续传 / 活跃任务管理 / SSE 响应构建 / LangGraph 流转换 ----

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Accel-Charset": "utf-8",
    "Content-Encoding": "identity",
    "Transfer-Encoding": "chunked",
}


class ActiveTaskManager:
    """活跃任务管理器：跟踪正在进行的 SSE 任务，支持取消"""
    _tasks: Dict[str, SSEService] = {}

    def register(self, thread_id: Any, sse: SSEService):
        self._tasks[str(thread_id)] = sse

    def unregister(self, thread_id: Any):
        self._tasks.pop(str(thread_id), None)

    def cancel(self, thread_id: Any) -> bool:
        sse = self._tasks.get(str(thread_id))
        if sse:
            sse.cancel()
            return True
        return False


# 全局单例，各端点共用
task_manager = ActiveTaskManager()


async def replay_events(
    thread_id: str, after_seq_id: int, sse: SSEService, event_store
) -> AsyncGenerator[str, None]:
    """断点续传：从事件存储补发遗漏的 SSE 事件"""
    events = await event_store.get_events_after(thread_id, after_seq_id)
    for doc in events:
        event = sse.next_event(
            event_type=doc["type"],
            status=doc["status"],
            data=doc["data"],
        )
        yield event.to_sse()


def build_sse_response(event_stream: AsyncGenerator[str, None]) -> StreamingResponse:
    """构建通用 SSE StreamingResponse"""
    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def graph_to_sse_events(
    stream: AsyncGenerator,
    sse: SSEService,
    thread_id: Any,
    node_map: Dict[str, dict],
    build_node_data: Callable[[str, dict], dict],
) -> AsyncGenerator[SSEEvent, None]:
    """
    通用 LangGraph stream → SSEEvent 生成器
    - node_map: 节点名 → {"type": ..., "status": ...} 映射
    - build_node_data: (node_name, state_update) → SSE data dict
    """
    task_manager.register(thread_id, sse)
    has_error = False
    try:
        async for node_name, state_update in stream:
            if sse.is_cancelled:
                yield sse_event(sse, "cancelled", "用户已取消")
                # 真正中断底层 graph 生成器，避免后台继续消耗 LLM token
                try:
                    await stream.aclose()
                except Exception:
                    pass
                return

            # 节点/流程错误（文生图/图生图 graph 异常时 yield "error", {"failed_node"}）
            if node_name == "error":
                has_error = True
                # 流程节点出错：只告知哪个步骤出问题，不透出原始节点名与错误原因
                if isinstance(state_update, dict) and state_update.get("failed_node") is not None:
                    step_name = get_step_name(state_update.get("failed_node")) or "当前步骤"
                    yield sse_event(
                        sse,
                        event_type="error",
                        status=f"「{step_name}」执行出错，请重试",
                        data={"threadId": thread_id},
                    )
                else:
                    # 兼容知识库等旧流程（graph 直接 yield {"msg"}）：仅透出提示语，不附堆栈
                    err_msg = (
                        state_update.get("msg", str(state_update))
                        if isinstance(state_update, dict) else str(state_update)
                    )
                    yield sse_event(
                        sse,
                        event_type="error",
                        status=f"执行出错：{err_msg}",
                        data={"messages": err_msg},
                    )
                continue

            # 节点开始信号（节点内 StreamWriter 在模型调用前发出，先展示"进行中"状态）
            if node_name == "node_start":
                payload = state_update if isinstance(state_update, dict) else {}
                node = payload.get("node", "")
                node_info = node_map.get(node, {})
                # 不透出原始节点名：未知节点统一用通用类型/中文提示
                data = {"messages": payload.get("messages") or "正在执行该步骤"}
                for key, value in payload.items():
                    if key in ("node", "messages"):
                        continue
                    data[key] = value
                yield sse_event(
                    sse,
                    event_type=node_info.get("type") or "step_process",
                    status=payload.get("status") or node_info.get("status") or "正在执行该步骤",
                    data=data,
                )
                continue

            # 人工中断
            if node_name == "__interrupt__":
                interrupt_payload = state_update[0].value if state_update else {}
                # 节点失败型中断（retry_node）：以 error 事件返回，仅告知哪个步骤失败，
                # 不透出原始节点名与错误原因；前端收到 error 后调用 /retry 从失败节点继续执行
                if interrupt_payload.get("retry_target"):
                    step_name = get_step_name(interrupt_payload.get("retry_target")) or "当前步骤"
                    yield sse_event(
                        sse,
                        event_type="error",
                        status=f"「{step_name}」执行失败，等待重试",
                        data={
                            "threadId": thread_id,
                            "retry_count": interrupt_payload.get("retry_count"),
                        },
                    )
                else:
                    # 真正的人工介入（如文生图的补充描述选择）：保持 human_in_the_loop
                    yield sse_interrupt(sse, thread_id, interrupt_payload)
                try:
                    await stream.aclose()
                except Exception:
                    pass
                return

            # 正常节点完成事件（未映射节点用通用类型/中文提示，不透出原始节点名）
            node_info = node_map.get(node_name, {})
            data = build_node_data(node_name, state_update)
            yield sse_event(
                sse,
                event_type=node_info.get("type") or "step_process",
                status=node_info.get("status") or "该步骤执行完成",
                data=data,
            )

        if not has_error:
            yield sse_done(sse)
    except asyncio.CancelledError:
        yield sse_event(sse, "cancelled", "用户已取消")
    except Exception:
        # 不透出内部错误原因，仅提示执行异常
        yield sse_event(sse, "error", "执行异常，请重试")
    finally:
        task_manager.unregister(thread_id)
