"""本地测试脚本：模拟连接服务器 MongoDB 并验证认证是否成功

用法（在项目根目录执行）：
    python scripts/test_mongo_connection.py
"""
import sys
from pathlib import Path

# 确保能导入 app 包（与项目根目录下的运行方式一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from app.config.settings import settings


def main() -> int:
    print("=" * 60)
    print("MongoDB 连接认证测试")
    print("=" * 60)
    print(f"连接地址 : {settings.mongodb_url}")
    print(f"数据库   : {settings.mongodb_db_name or '(未配置)'}")
    print(f"账号     : {settings.mongodb_account or '(未配置)'}")
    print(f"密码     : {'***' if settings.mongodb_password else '(未配置)'}")
    print(f"认证参数 : {settings.mongodb_conn_kwargs or '(无，走本地免密)'}")
    print("-" * 60)

    try:
        from pymongo import MongoClient

        client = MongoClient(
            settings.mongodb_url,
            **settings.mongodb_conn_kwargs,
            serverSelectionTimeoutMS=10_000,
        )

        # 1) ping 验证服务可达
        client.admin.command("ping")
        print("[OK] 服务器可达 (ping 成功)")

        # 2) connectionStatus 验证认证用户
        status = client.admin.command("connectionStatus")
        auth_users = status.get("authInfo", {}).get("authenticatedUsers", [])
        if auth_users:
            for u in auth_users:
                print(f"[OK] 认证成功，登录用户: {u.get('user')} @ {u.get('db')}")
        else:
            # 部分版本返回结构不同，尝试从 serverStatus 获取
            conn_status = client.admin.command("connectionStatus").get("authInfo", {})
            print("[WARN] connectionStatus 未返回用户信息，结构: %s" % conn_status)

        # 3) 列出当前用户可见的数据库，确认权限
        db_names = client.list_database_names()
        print(f"[OK] 可见数据库: {db_names}")

        # 4) 测试目标库可读写
        db = client[settings.mongodb_db_name or "admin"]
        test_col = db["_conn_test"]
        test_col.insert_one({"probe": True})
        doc = test_col.find_one({"probe": True})
        test_col.delete_many({"probe": True})
        print(f"[OK] 数据库 [{db.name}] 读写正常 (探针文档已清理)")

        client.close()
        print("-" * 60)
        print("结果: 连接认证成功")
        return 0

    except OperationFailure as e:
        print(f"[FAIL] 认证失败 (OperationFailure): {e}")
        print("提示: 请检查 MONGODB_ACCOUNT / MONGODB_PASSWORD 是否正确")
        return 1
    except ServerSelectionTimeoutError as e:
        print(f"[FAIL] 无法连接服务器 (ServerSelectionTimeoutError): {e}")
        print("提示: 请检查 MONGODB_URL、网络/防火墙是否可达，账号密码是否有效")
        return 1
    except Exception as e:
        print(f"[FAIL] 未知错误: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
