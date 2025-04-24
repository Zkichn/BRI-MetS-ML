import urllib.request
import os

url = "https://github.com/Zkichn/BRI-MetS-ML/releases/download/v1.0/trained_models.zip"
save_path = "./saved_models/trained_models.zip"

os.makedirs("./saved_models", exist_ok=True)
print("🔽 正在下载模型...")
urllib.request.urlretrieve(url, save_path)
print(f"✅ 模型已保存至: {save_path}")
