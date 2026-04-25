from fastapi import FastAPI, HTTPException
from app.models import ExecuteRequest, ExecuteResponse, ExecuteProjectRequest
from app.runner import run_python_code, run_python_project

app = FastAPI(
    title="Code Execution API",
    description="An API to execute code snippets and return their output.",
    version="1.0.0",
)


@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    if request.language != "python":
        raise HTTPException(
            status_code=400, detail=f"Unsupported language: {request.language}"
        )
    return run_python_code(
        code=request.code,
        requirements=request.requirements,
        python_version=request.python_version,
        timeout_seconds=request.timeout_seconds,
    )


@app.post("/execute_project", response_model=ExecuteResponse)
def execute_project(request: ExecuteProjectRequest):
    return run_python_project(
        files=request.files,
        command=request.command,
        requirements=request.requirements,
        python_version=request.python_version,
        timeout_seconds=request.timeout_seconds,
    )
