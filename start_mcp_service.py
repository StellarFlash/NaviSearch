"""
start_mcp_service.py

MCP服务启动脚本
"""
from MCP_Visitor import MCPVisitorService
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
mcp_service = MCPVisitorService()

@app.post("/mcp")
async def handle_mcp_request(request: Request):
    """处理MCP请求"""
    request_data = await request.json()
    return mcp_service.process_request(request_data)

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """启动MCP服务"""
    print(f"Starting MCP service on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()