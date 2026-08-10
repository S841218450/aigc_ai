import requests

from app.api.cloud_base_client import cloudbase


def generate_image(prompt):
	"""调用生图云函数"""
	url = f"{cloudbase.base_url}"
	headers = cloudbase.headers
	payload = {
		"prompt": prompt
	}

	try:
		response = requests.post(url, headers=headers, json=payload)
		response.raise_for_status()
		result = response.json()

		if result.get("success"):
			# 生成成功
			print("生成成功!")
			print(f"图片URL: {result.get('imageUrl')}")
			print(f"优化后的提示词: {result.get('revised_prompt')}")

			# 使用图片
			# 注意：图片URL有效期为24小时，请及时保存或转存
			return result
		else:
			# 生成失败
			print(f"生成失败: {result.get('code')} {result.get('message')}")
			return None
	except Exception as e:
		print(f"调用失败: {e}")
		return None

# 使用示例
if __name__ == "__main__":
	result = generate_image("一只可爱的猫咪在阳光下玩耍")
	if result:
		print(f"图片URL: {result.get('imageUrl')}")