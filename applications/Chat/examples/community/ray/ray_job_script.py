import sys

from ray.job_submission import JobSubmissionClient


def main(api_server_endpoint="http://127.0.0.1:8265"):
    client = JobSubmissionClient(api_server_endpoint)
    client.submit_job(
        entrypoint="python experimental/ray/train_prompts_on_ray.py --strategy colossalai_zero2 --prompt_csv_url https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv",
        runtime_env={
            "working_dir": "applications/Chat",
            "pip": [
                "torch==1.13.1",
                "transformers>=4.20.1",
                "datasets",
                "loralib",
                "colossalai>=0.2.4",
                "langchain",
                "tokenizers",
                "fastapi",
                "sse_starlette",
                "wandb",
                "sentencepiece",
                "gpustat",
            ],
        },
    )


if __name__ == "__main__":
    main(sys.argv[1])
