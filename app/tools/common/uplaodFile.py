from app.api.java_client import API

async def uploadFile(file):
  """
  上传文件
  """
  if not file:
    return None
  res = await API.post("/file/upload", files={"file": file})
  return res.json()



