import asyncio
import hashlib
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config.settings import settings
from app.core.agents.model_factory import get_model
from app.utils.logger_handle import logger


# ---------------------------------------------------------------------------
# 元数据过滤工具函数（给 BM25 检索用，语法对齐 Chroma where filter）
#   支持：
#     等值    : {"folder_id": 123}
#     $in     : {"folder_id": {"$in": [123,124]}}
#     $nin/$gt/$gte/$lt/$lte/$ne: {"num__price": {"$lte": 300}}
#     $and/$or: [ {...}, {...} ]
# ---------------------------------------------------------------------------

def _metadata_matches(metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
    if not filter_dict:
        return True
    md = metadata or {}
    for key, expected in filter_dict.items():
        if key == "$and":
            if not isinstance(expected, list) or not all(isinstance(c, dict) for c in expected):
                return False
            if not all(_metadata_matches(md, cond) for cond in expected):
                return False
            continue
        if key == "$or":
            if not isinstance(expected, list) or not all(isinstance(c, dict) for c in expected):
                return False
            if not any(_metadata_matches(md, cond) for cond in expected):
                return False
            continue
        actual = md.get(key)
        if isinstance(expected, dict):
            # op: {"$in": [...]} / {"$lte": 300} / {"$ne": "x"}
            for op, op_val in expected.items():
                if op == "$in":
                    if isinstance(op_val, (list, tuple, set)):
                        if actual not in op_val:
                            return False
                    else:
                        return False
                elif op == "$nin":
                    if isinstance(op_val, (list, tuple, set)):
                        if actual in op_val:
                            return False
                    else:
                        return False
                elif op == "$eq":
                    if actual != op_val:
                        return False
                elif op == "$ne":
                    if actual == op_val:
                        return False
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    if actual is None or op_val is None:
                        return False
                    try:
                        if op == "$gt" and not (actual > op_val):
                            return False
                        if op == "$gte" and not (actual >= op_val):
                            return False
                        if op == "$lt" and not (actual < op_val):
                            return False
                        if op == "$lte" and not (actual <= op_val):
                            return False
                    except TypeError:
                        return False
                else:
                    # 未知操作符：保守跳过，不匹配
                    return False
        else:
            if actual != expected:
                return False
    return True


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self._initialized = False

    def add_documents(self, docs: List[Dict[str, Any]]):
        for doc in docs:
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            self.docs.append(doc)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lens.append(len(tokens))

        if self.doc_lens:
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        self._compute_idf()
        self._initialized = True

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return tokens

    def _compute_idf(self):
        n_docs = len(self.docs)
        df = Counter()
        for freq in self.doc_freqs:
            for term in freq.keys():
                df[term] += 1
        for term, freq in df.items():
            self.idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 10,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        if not self._initialized or not self.docs:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        # 支持 metadata_filter：先过滤再算分，省大量计算
        if metadata_filter:
            candidate_indices = [
                i for i, d in enumerate(self.docs)
                if _metadata_matches(d.get("metadata", {}) or {}, metadata_filter)
            ]
        else:
            candidate_indices = range(len(self.docs))

        for i in candidate_indices:
            doc_freq = self.doc_freqs[i]
            doc_len = self.doc_lens[i]
            doc = self.docs[i]
            score = 0.0
            for token in query_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                idf = self.idf.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class VectorStoreTool:
    def __init__(self):
        self.embeddings = get_model('embedding')
        self.vector_store = Chroma(
            persist_directory=settings.chroma_persist_directory,
            embedding_function=self.embeddings
        )
        self.bm25_retriever = BM25Retriever()
        self._bm25_built = False

    async def _ensure_bm25_built(self):
        if self._bm25_built:
            return
        try:
            collection = self.vector_store._collection
            all_docs = collection.get()
            docs = []
            for i, content in enumerate(all_docs.get("documents", [])):
                metadata = all_docs.get("metadatas", [])[i] if all_docs.get("metadatas") else {}
                docs.append({
                    "content": content,
                    "metadata": metadata,
                    "id": all_docs.get("ids", [])[i] if all_docs.get("ids") else str(i)
                })
            self.bm25_retriever.add_documents(docs)
            self._bm25_built = True
        except Exception as e:
            print(f"[BM25] 构建索引失败: {e}")

    async def vector_search(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        k = params.get("k", 50) if params else 50
        filter_metadata = params.get("filter", {}) if params else {}

        results = await self.vector_store.asimilarity_search_with_score(
            query,
            k=k,
            filter=filter_metadata if filter_metadata else None
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "retrieval_method": "vector"
            }
            for doc, score in results
        ]

    async def bm25_search(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        await self._ensure_bm25_built()
        k = params.get("k", 50) if params else 50
        filter_metadata = params.get("filter", {}) if params else {}
        results = self.bm25_retriever.search(
            query, top_k=k, metadata_filter=filter_metadata if filter_metadata else None
        )
        return [
            {
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": float(score),
                "retrieval_method": "bm25"
            }
            for doc, score in results
        ]

    async def hybrid_search(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        vector_weight = params.get("vector_weight", 0.6) if params else 0.6
        bm25_weight = params.get("bm25_weight", 0.4) if params else 0.4
        top_k = params.get("k", 20) if params else 20

        vector_docs = await self.vector_search(query, params)
        bm25_docs = await self.bm25_search(query, params)

        if not vector_docs and not bm25_docs:
            return []

        doc_scores: Dict[str, Dict[str, Any]] = {}

        for doc in vector_docs:
            doc_id = doc["metadata"].get("chunk_id") or doc["metadata"].get("id") or doc["content"][:50]
            normalized_score = self._normalize_score(doc["score"], "vector")
            doc_scores[doc_id] = {
                **doc,
                "hybrid_score": normalized_score * vector_weight,
                "vector_score": doc["score"],
                "bm25_score": 0.0
            }

        for doc in bm25_docs:
            doc_id = doc["metadata"].get("chunk_id") or doc["metadata"].get("id") or doc["content"][:50]
            normalized_score = self._normalize_score(doc["score"], "bm25")
            if doc_id in doc_scores:
                doc_scores[doc_id]["hybrid_score"] += normalized_score * bm25_weight
                doc_scores[doc_id]["bm25_score"] = doc["score"]
                doc_scores[doc_id]["retrieval_method"] = "hybrid"
            else:
                doc_scores[doc_id] = {
                    **doc,
                    "hybrid_score": normalized_score * bm25_weight,
                    "vector_score": 0.0,
                    "bm25_score": doc["score"]
                }

        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        return sorted_docs[:top_k]

    def _normalize_score(self, score: float, method: str) -> float:
        if method == "vector":
            return max(0.0, min(1.0, 1.0 - score)) if score > 0 else 1.0
        elif method == "bm25":
            return max(0.0, min(1.0, score / 10.0)) if score > 0 else 0.0
        return 0.0

    async def rerank(self, query: str, docs: List[Dict[str, Any]], params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        top_k = params.get("rerank_k", 5) if params else 5
        if not docs:
            return []

        scored_docs = []
        for doc in docs:
            content = doc["content"]
            query_terms = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))
            content_terms = set(re.findall(r'[\w\u4e00-\u9fff]+', content.lower()))

            if not query_terms:
                relevance = 0.0
            else:
                overlap = query_terms & content_terms
                relevance = len(overlap) / len(query_terms)

            hybrid_score = doc.get("hybrid_score", doc.get("score", 0))
            final_score = 0.7 * hybrid_score + 0.3 * relevance

            scored_doc = {**doc, "rerank_score": final_score}
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]

    async def search(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        strategy = params.get("retrieval_strategy", "hybrid") if params else "hybrid"

        if strategy == "bm25":
            return await self.bm25_search(query, params)
        elif strategy == "vector":
            return await self.vector_search(query, params)
        else:
            return await self.hybrid_search(query, params)

    @staticmethod
    def compute_content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def check_duplicate(self, content: str, semantic_threshold: float = 0.85) -> Dict[str, Any]:
        content_hash = self.compute_content_hash(content)

        try:
            collection = self.vector_store._collection
            existing = collection.get(where={"content_hash": content_hash})
            if existing.get("ids"):
                return {
                    "is_duplicate": True,
                    "duplicate_type": "exact",
                    "existing_id": existing["ids"][0]
                }
        except Exception:
            pass

        try:
            results = await self.vector_store.asimilarity_search_with_score(content, k=1)
            if results:
                _, score = results[0]
                similarity = 1.0 - score if score >= 0 else 1.0
                if similarity >= semantic_threshold:
                    return {
                        "is_duplicate": True,
                        "duplicate_type": "semantic",
                        "similarity": similarity
                    }
        except Exception:
            pass

        return {"is_duplicate": False}

    @staticmethod
    def build_context(docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", metadata.get("doc_name", "未知来源"))
            section = metadata.get("section", metadata.get("chapter", ""))
            update_time = metadata.get("update_time", metadata.get("created_at", ""))
            permission = metadata.get("permission", "")

            header = f"[片段{i}]"
            if source:
                header += f" 来源: {source}"
            if section:
                header += f" 章节: {section}"
            if update_time:
                header += f" 更新时间: {update_time}"
            if permission:
                header += f" 权限: {permission}"

            context_parts.append(f"{header}\n{doc['content']}")

        return "\n\n".join(context_parts)

    @staticmethod
    def extract_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen_docs = set()

        for doc in docs:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id", metadata.get("source", ""))

            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            source_info = {
                "doc_name": metadata.get("doc_name", metadata.get("source", "未知文档")),
                "update_time": metadata.get("update_time", metadata.get("created_at", "")),
                "section": metadata.get("section", metadata.get("chapter", "")),
                "permission": metadata.get("permission", ""),
                "page": metadata.get("page", ""),
                "doc_id": doc_id,
                "score": doc.get("rerank_score", doc.get("hybrid_score", doc.get("score", 0)))
            }
            sources.append(source_info)

        return sources

    # ========================
    # 向量写入 / 删除 / 管理
    # ========================

    def _reset_bm25_cache(self):
        """向量库写入/删除后强制重置 BM25 索引，下次检索时从 Chroma 最新数据重建"""
        self._bm25_built = False
        self.bm25_retriever = BM25Retriever()

    async def upsert_chunks(self, chunks: List[Dict[str, Any]], replace_existing_doc: bool = True) -> Dict[str, Any]:
        """
        批量 upsert 文档 chunk 到 Chroma 向量库。

        Args:
            chunks: DocumentProcessor.process_document() 返回的 chunks，
                    每项必须包含 {"content": str, "metadata": {"chunk_id": str, "doc_id": str, ...}}
            replace_existing_doc: 若 doc_id 已存在，是否先删除旧 chunk 再插入（默认 True，保证幂等）

        Returns:
            {"inserted_count": int, "skipped_duplicates": int, "doc_id": str}
        """
        if not chunks:
            return {"inserted_count": 0, "skipped_duplicates": 0, "doc_id": ""}

        doc_ids_in_chunks = list({c["metadata"].get("doc_id", "") for c in chunks if c.get("metadata")})
        current_doc_id = doc_ids_in_chunks[0] if len(doc_ids_in_chunks) == 1 else ""

        collection = self.vector_store._collection

        # 幂等：同一 doc_id 先删旧 chunk 再插入新的，避免新旧混合
        if replace_existing_doc and current_doc_id:
            try:
                await asyncio.to_thread(
                    collection.delete,
                    where={"doc_id": current_doc_id}
                )
            except Exception as e:
                print(f"[VectorStore] upsert 删除旧 chunk 失败(doc_id={current_doc_id}): {e}")

        texts = []
        metadatas = []
        ids = []
        seen_hashes = set()
        seen_ids = set()
        skipped_duplicates = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {}) or {}
            chunk_id = metadata.get("chunk_id") or metadata.get("id") or hashlib.md5(content.encode()).hexdigest()[:12]
            content_hash = metadata.get("content_hash") or hashlib.sha256(content.encode()).hexdigest()

            # 同一批次内基于 content_hash 再次去重（防止 processor 有漏网之鱼）
            if content_hash in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(content_hash)

            # 兜底：chunk_id 批次内重复时，用内容 hash 重新生成唯一 id
            # （内容已被 content_hash 去重，内容不同则重生成的 id 必不同）
            if chunk_id in seen_ids:
                chunk_id = hashlib.md5(f"{chunk_id}_{content_hash}".encode()).hexdigest()[:12]
                metadata = {**metadata, "chunk_id": chunk_id}
            seen_ids.add(chunk_id)

            if "content_hash" not in metadata:
                metadata["content_hash"] = content_hash

            texts.append(content)
            metadatas.append(metadata)
            ids.append(chunk_id)

        # add_texts 内部会调 embedding API，可能耗时，包 asyncio.to_thread 避免阻塞事件循环
        # 上游 embedding 服务偶发 transient 错误（400 InternalError），指数退避重试 3 次
        for attempt in range(1, 4):
            try:
                await asyncio.to_thread(
                    self.vector_store.add_texts,
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                break
            except Exception as e:
                if attempt < 3:
                    wait = 2 ** attempt  # 2s / 4s
                    logger.warning(f"[VectorStore] embedding 第 {attempt} 次失败，{wait}s 后重试: {e}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"[VectorStore] embedding 重试 3 次仍失败: {e}")
                    raise

        self._reset_bm25_cache()

        return {
            "inserted_count": len(texts),
            "skipped_duplicates": skipped_duplicates,
            "doc_id": current_doc_id,
        }

    async def delete_by_doc_id(self, doc_id: str) -> Dict[str, Any]:
        """
        按 doc_id 删除向量库中所有关联 chunk + 对应的元数据。
        与 Java 端文件删除保持强一致。
        """
        if not doc_id:
            return {"deleted": False, "reason": "doc_id 为空"}

        collection = self.vector_store._collection

        try:
            # 先查一下到底有多少 chunk 要删（用于日志/返回）
            existing = await asyncio.to_thread(
                collection.get,
                where={"doc_id": doc_id},
                include=[],
            )
            chunk_ids = existing.get("ids", []) or []
            deleted_count = len(chunk_ids)

            if deleted_count > 0:
                await asyncio.to_thread(
                    collection.delete,
                    where={"doc_id": doc_id},
                )
                self._reset_bm25_cache()

            return {"deleted": True, "deleted_chunk_count": deleted_count, "doc_id": doc_id}
        except Exception as e:
            return {"deleted": False, "reason": f"向量库删除异常: {str(e)}", "doc_id": doc_id}

    async def delete_by_doc_ids(self, doc_ids: List[str]) -> Dict[str, Any]:
        """批量按 doc_ids 删除（使用 Chroma where $in 操作符，单次请求搞定，省 N 次 HTTP）"""
        if not doc_ids:
            return {"deleted": False, "reason": "doc_ids 为空"}

        collection = self.vector_store._collection

        try:
            existing = await asyncio.to_thread(
                collection.get,
                where={"doc_id": {"$in": doc_ids}},
                include=[],
            )
            chunk_ids = existing.get("ids", []) or []
            deleted_chunk_count = len(chunk_ids)

            if deleted_chunk_count > 0:
                await asyncio.to_thread(
                    collection.delete,
                    where={"doc_id": {"$in": doc_ids}},
                )
                self._reset_bm25_cache()

            return {
                "deleted": True,
                "deleted_doc_count": len(doc_ids),
                "deleted_chunk_count": deleted_chunk_count,
            }
        except Exception as e:
            return {"deleted": False, "reason": f"批量删除向量库异常: {str(e)}"}

    async def list_doc_ids(self, limit: int = 10000) -> List[str]:
        """列出向量库中所有已知的 doc_id（用于对账，不要大 limit 扫库）"""
        collection = self.vector_store._collection
        try:
            data = await asyncio.to_thread(
                collection.get,
                include=[],  # 只拿 ids 和 metadatas，不要文本和 embedding，省内存
                limit=limit,
            )
            metadatas = data.get("metadatas", []) or []
            doc_ids = {m.get("doc_id") for m in metadatas if isinstance(m, dict) and m.get("doc_id")}
            return sorted(doc_ids)
        except Exception as e:
            print(f"[VectorStore] list_doc_ids 失败: {e}")
            return []

    async def clear(self) -> Dict[str, Any]:
        """清空整个向量库（⚠️ 危险操作，仅用于排障/重建）"""
        collection = self.vector_store._collection
        try:
            existing = await asyncio.to_thread(collection.get, include=[])
            total = len(existing.get("ids", []) or [])
            if total > 0:
                await asyncio.to_thread(collection.delete)
                self._reset_bm25_cache()
            return {"cleared": True, "deleted_chunk_count": total}
        except Exception as e:
            return {"cleared": False, "reason": f"清空向量库异常: {str(e)}"}
