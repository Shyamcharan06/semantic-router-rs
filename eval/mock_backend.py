"""A stand-in OpenAI-compatible backend for local development and eval runs.

It doesn't call any real LLM -- it just echoes back which `model` the router
sent it, so you can verify routing decisions end-to-end without needing API
keys for five different providers. Point config/routes.yaml's category
backends at this (default: http://localhost:9000) to run the whole stack
locally.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    last_user_message = next(
        (m["content"] for m in reversed(req.messages) if m.get("role") == "user"),
        "",
    )
    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "model": req.model,
        "echo_model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[mock response from '{req.model}'] You said: {last_user_message[:80]}",
                },
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
