import os
import traceback
import json
import asyncio
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

# Import FastMCP and Context from MCP SDK
from mcp.server.fastmcp import Context, FastMCP, Image # Include Image if needed later, though not directly used in Visitor currently

# 导入 NaviSearchVisitor 类及其依赖
# 请确保您的 SearchEngine.py, SemanticTagger.py, utils.py 可导入
# 为了方便，我将提供的 NaviSearchVisitor 类代码和必要的依赖占位符包含在下面
# --- Start of NaviSearchVisitor Class Code (adapted for lifespan) ---
from pymilvus import MilvusClient, Collection, Connections

from Visitor import NaviSearchVisitor  # 导入 NaviSearchVisitor 类及其依赖

DEFAULT_COLLECTION_NAME = "navi_search_collection"
DEFAULT_MILVUS_TOKEN = "root:Milvus"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

@dataclass
class AppContext:
    """Holds application-specific context, like the NaviSearchVisitor instance."""
    visitor: NaviSearchVisitor

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manages the lifecycle of the NaviSearchVisitor."""
    print("Starting app lifespan...")

    # Load configuration (ideally from environment or UVX config)
    milvus_host = os.environ.get("MILVUS_HOST", DEFAULT_MILVUS_HOST)
    milvus_port = os.environ.get("MILVUS_PORT", DEFAULT_MILVUS_PORT)
    milvus_token = os.environ.get("MILVUS_TOKEN", DEFAULT_MILVUS_TOKEN)
    collection_name = os.environ.get("NAVI_SEARCH_COLLECTION", DEFAULT_COLLECTION_NAME)

    # Initialize the Visitor instance
    visitor_instance = NaviSearchVisitor(
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_token=milvus_token
    )

    # Connect Milvus during startup
    connect_status = visitor_instance.connect_milvus()
    if connect_status['status'] in ['error', 'warning']:
         print(f"Visitor initialization warning/error: {connect_status['message']}")
         # Depending on requirements, you might raise an exception here to prevent startup on error
         # raise ConnectionError(f"Failed to initialize NaviSearchVisitor: {connect_status['message']}")

    try:
        # Yield the context containing the initialized visitor
        print("App lifespan started successfully. Yielding context.")
        yield AppContext(visitor=visitor_instance)
    finally:
        # Clean up resources on shutdown
        print("Stopping app lifespan. Disconnecting Milvus...")
        visitor_instance.disconnect_milvus()
        print("App lifespan stopped.")


# Create a named FastMCP server instance with the defined lifespan
mcp = FastMCP("NaviSearchVisitorService", lifespan=app_lifespan)

# --- Define MCP Tools mapping to NaviSearchVisitor methods ---

@mcp.tool()
async def set_target_collection(ctx: Context, collection_name: str) -> Dict[str, Any]:
    """
    MCP Tool to set the target collection name for search operations.
    Args:
        collection_name (str): The name of the Milvus collection to target.
    Returns:
        Dict[str, Any]: Operation status and message.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print(f"Tool 'set_target_collection' called with collection_name='{collection_name}'")
    return visitor.set_target_collection_name(collection_name)

@mcp.tool()
async def get_current_collection(ctx: Context) -> str:
    """
    MCP Tool to get the current target collection name.
    Returns:
        str: The name of the current target collection.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print("Tool 'get_current_collection' called.")
    return visitor.get_current_target_collection_name()

@mcp.tool()
async def add_session_tags(ctx: Context, tag_content: str) -> Dict[str, Any]:
    """
    MCP Tool to add active session tags for filtering future searches (standard mode).
    Tags should be comma-separated.
    Args:
        tag_content (str): A comma-separated string of tags to add.
    Returns:
        Dict[str, Any]: Operation status, message, and current active tags.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print(f"Tool 'add_session_tags' called with tag_content='{tag_content}'")
    return visitor.add_tags(tag_content)

@mcp.tool()
async def clear_session_tags(ctx: Context) -> Dict[str, Any]:
    """
    MCP Tool to clear all active session tags.
    Returns:
        Dict[str, Any]: Operation status, message, and empty active tags list.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print("Tool 'clear_session_tags' called.")
    return visitor.clear_tags()

@mcp.tool()
async def get_active_session_tags(ctx: Context) -> List[str]:
    """
    MCP Tool to get the current list of active session tags.
    Returns:
        List[str]: List of active tags.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print("Tool 'get_active_session_tags' called.")
    return visitor.get_active_tags()

@mcp.tool()
async def search(ctx: Context,
                 query_text: str,
                 filter_tags: Optional[List[str]] = None,
                 mode: str = "standard",
                 retrieval_top_k: int = 20,
                 rerank_strategy: Optional[str] = "ranking",
                 rerank_doc_top_k: Optional[int] = 5,
                 rerank_tags_top_k: Optional[int] = 10,
                 agentic_rerank_stop_size: Optional[int] = 3,
                 agentic_max_iterations: Optional[int] = 10) -> Dict[str, Any]:
    """
    MCP Tool to perform a search operation.
    Args:
        query_text (str): The user's search query.
        filter_tags (List[str], optional): Temporary filter tags for this search (overrides session tags in standard mode).
        mode (str): Search mode ("standard" or "agentic").
        retrieval_top_k (int): Number of records to retrieve initially.
        rerank_strategy (str, optional): Reranking strategy ("ranking" or "filtering").
        rerank_doc_top_k (int, optional): Number of documents to return after reranking (standard mode).
        rerank_tags_top_k (int, optional): Number of recommended tags to return.
        agentic_rerank_stop_size (int, optional): Stop size for agentic search.
        agentic_max_iterations (int, optional): Max iterations for agentic search.
    Returns:
        Dict[str, Any]: Search results including ranked records and recommended tags.
    """
    visitor = ctx.request_context.lifespan_context.visitor
    print(f"Tool 'search' called with query='{query_text}', mode='{mode}'")
    if filter_tags is not None:
         print(f"  Temporary filter_tags: {filter_tags}")
    # Pass all explicit arguments and potentially others via kwargs if needed by underlying method
    return await visitor.search(
        query_text=query_text,
        filter_tags=filter_tags,
        mode=mode,
        retrieval_top_k=retrieval_top_k,
        rerank_strategy=rerank_strategy,
        rerank_doc_top_k=rerank_doc_top_k,
        rerank_tags_top_k=rerank_tags_top_k,
        agentic_rerank_stop_size=agentic_rerank_stop_size,
        agentic_max_iterations=agentic_max_iterations,
        # Pass other kwargs if any, though the tool definition is explicit
    )


# --- MCP Server Entry Point ---

# This block is the standard way to run a FastMCP server.
# UVX is expected to use this or similar logic (e.g., calling mcp.run() or using mcp.sse_app()).
if __name__ == "__main__":
    print("Starting MCP_VisitorService using mcp.run()...")
    # mcp.run() will automatically use the lifespan manager defined with the FastMCP instance
    mcp.run()

    # Alternatively, for development with the inspector:
    # To run with `mcp dev MCP_Visitor.py`: the FastMCP instance `mcp` is automatically discovered.
    # You might need `--with .` if Search/Tagger/utils are local.
    # Example command: mcp dev MCP_Visitor.py --with .