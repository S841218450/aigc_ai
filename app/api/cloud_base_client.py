import os
import requests
from dotenv import load_dotenv

load_dotenv()

class CloudBaseClient:
	def __init__(self):
		self.env_id = os.getenv("CLOUDBASE_ENV_ID")
		self.access_token = os.getenv("CLOUDBASE_ACCESS_TOKEN")
		self.base_url = f"https://{self.env_id}.api.tcloudbasegateway.com/v1/ai/cloudbase"
		self.headers = {
			"Content-Type": "application/json",
			"Accept": "application/json",
			"Authorization": f"Bearer {self.access_token}"
		}

	def request(self, method, path, **kwargs):
		"""
		统一的HTTP请求方法

		Args:
			method: 请求方法 (GET, POST, PUT, PATCH, DELETE)
			path: API路径 (如 /v1/rdb/rest/table_name)
			**kwargs: 其他请求参数 (json, params, headers等)

		Returns:
			响应数据或None
		"""
		url = f"{self.base_url}{path}"
		headers = self.headers.copy()

		# 允许自定义headers
		if "headers" in kwargs:
			headers.update(kwargs.pop("headers"))

		try:
			response = requests.request(method, url, headers=headers, **kwargs)
			response.raise_for_status()

			# 如果响应为空，返回True表示成功
			if not response.content:
				return True

			return response.json()
		except requests.exceptions.RequestException as e:
			print(f"请求失败: {e}")
			return None

cloudbase = CloudBaseClient()