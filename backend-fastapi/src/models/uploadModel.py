from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load credentials from your .env file
load_dotenv()

repo_id = os.getenv("HF_REPO_ID")
token = os.getenv("HF_TOKEN")

print(f"Uploading artifacts to Hugging Face repo: {repo_id}...")

api = HfApi()
api.upload_folder(
    folder_path="../../outputs",  
    repo_id=repo_id,
    repo_type="model",
    token=token
)

print("Upload complete! Model files are now live on Hugging Face.")