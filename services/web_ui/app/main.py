import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class RunFlowRequest(BaseModel):
    config_path: str
    model_type: str

@app.post("/run_flow/")
async def run_flow(request: RunFlowRequest):
    try:
        # Đặt đường dẫn tới môi trường Conda
        conda_env_name = "computer-viz-dl"
        command = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env_name} && python run_flow.py --config {request.config_path} --model_type {request.model_type}"
        ]

        # Thực thi lệnh
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            shell=True
        )

        # Kiểm tra kết quả
        if process.returncode != 0:
            raise Exception(process.stderr)

        return {
            "status": "success",
            "output": process.stdout
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
