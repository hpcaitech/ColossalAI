from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")
client.submit_job(
    entrypoint=
    "python examples/train_prompts_on_ray.py --strategy colossalai_zero2 --prompt_csv_url https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv",
    runtime_env={
        "working_dir":
            "../",
        "pip": [
            "torch==1.13.1", "transformers>=4.20.1", "datasets", "loralib", "colossalai>=0.2.4", "langchain",
            "tokenizers", "fastapi", "sse_starlette", "wandb", "sentencepiece", "gpustat"
        ]
    })

# Use this script with 'python train_prompts_on_ray_job_submission_script' on your Ray cluster.
