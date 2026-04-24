from huggingface_hub import HfApi, upload_folder

api = HfApi()

upload_folder(
    folder_path="./outputs/best_model/",
    repo_id="Lris47/mllm_stage1",
    repo_type="model"
)