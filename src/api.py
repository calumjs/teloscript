import asyncio
import time
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

import sys
import os

try:
    from .models import (
        MCPAgentConfig, AgentGoal, AgentStatus, StatusUpdate, 
        AgentRequest, MCPServerConfig, AgentResponse,
        PurposeEndpointConfig, PurposeEndpointRequest, PurposeEndpointResponse
    )
    from .mcp_agent import LLMPoweredMCPAgent
    from .orchestrator import WorkflowOrchestrator
    from .purpose_manager import PurposeEndpointManager
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from models import (
        MCPAgentConfig, AgentGoal, AgentStatus, StatusUpdate, 
        AgentRequest, MCPServerConfig, AgentResponse,
        PurposeEndpointConfig, PurposeEndpointRequest, PurposeEndpointResponse
    )
    from mcp_agent import LLMPoweredMCPAgent
    from orchestrator import WorkflowOrchestrator
    from purpose_manager import PurposeEndpointManager

load_dotenv()

app = FastAPI(
    title="TELOSCRIPT API",
    description="Purposeful Agent Orchestration System - REST API for coordinating autonomous MCP agents toward goals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = WorkflowOrchestrator()
purpose_manager = PurposeEndpointManager()
status_streams: Dict[str, List[StatusUpdate]] = {}
active_streams: Dict[str, bool] = {}
orchestration_data: Dict[str, Dict] = {}
orchestration_subscribers: List[asyncio.Queue] = []

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "mcp_configs.json"
mcp_configs: Dict[str, Dict] = {}

def load_mcp_configs() -> Dict[str, Dict]:
    """Load MCP configurations from JSON file"""
    global mcp_configs
    
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                mcp_configs = json.load(f)
                logger.info(f"Loaded {len(mcp_configs)} MCP configurations from {CONFIG_FILE}")
        else:
            default_configs = {
                "filesystem": {
                    "name": "Filesystem Server (Current Directory)",
                    "description": "Access files in the current project directory",
                    "config": {
                        "name": "filesystem",
                        "command": "@modelcontextprotocol/server-filesystem",
                        "args": ["."],
                        "transport": "stdio"
                    }
                },
                "filesystem-home": {
                    "name": "Filesystem Server (Home Directory)",
                    "description": "Access files in your home directory",
                    "config": {
                        "name": "filesystem-home",
                        "command": "@modelcontextprotocol/server-filesystem",
                        "args": ["/home/agent"],
                        "transport": "stdio"
                    }
                },
                "filesystem-documents": {
                    "name": "Filesystem Server (Documents)",
                    "description": "Access files in your Documents folder",
                    "config": {
                        "name": "filesystem-documents",
                        "command": "@modelcontextprotocol/server-filesystem",
                        "args": ["/home/agent/Documents"],
                        "transport": "stdio"
                    }
                },
                "brave-search": {
                    "name": "Brave Search",
                    "description": "Web search capabilities",
                    "config": {
                        "name": "brave-search",
                        "command": "@modelcontextprotocol/server-brave-search",
                        "args": [],
                        "env": {"BRAVE_API_KEY": "your-api-key-here"},
                        "transport": "stdio"
                    }
                },
                "github": {
                    "name": "GitHub Integration",
                    "description": "Access GitHub repositories and data",
                    "config": {
                        "name": "github",
                        "command": "@modelcontextprotocol/server-github",
                        "args": [],
                        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"},
                        "transport": "stdio"
                    }
                },
                "sqlite": {
                    "name": "SQLite Database",
                    "description": "Access SQLite databases",
                    "config": {
                        "name": "sqlite",
                        "command": "@modelcontextprotocol/server-sqlite",
                        "args": ["/path/to/database.db"],
                        "transport": "stdio"
                    }
                },
                "memory": {
                    "name": "Memory Store",
                    "description": "Persistent memory storage for agents",
                    "config": {
                        "name": "memory",
                        "command": "@modelcontextprotocol/server-memory",
                        "args": [],
                        "transport": "stdio"
                    }
                },
                "puppeteer": {
                    "name": "Puppeteer",
                    "description": "Use puppeteer to browse the internet",
                    "config": {
                        "name": "puppeteer",
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-puppeteer"
                        ],
                        "transport": "stdio"
                    }
                },
                "playwright": {
                    "command": "npx",
                    "name": "Playwright",
                    "description": "Use Playwright to browse the internet",
                    "config": {
                        "name": "playwright",
                        "command": "npx",
                        "args": [
                            "@playwright/mcp@latest"
                        ],
                        "transport": "stdio"
                    }
                },
                "tavily": {
                    "name": "Tavily",
                    "description": "Search for up to date information online",
                    "config": {
                        "name": "tavily-mcp",
                        "command": "npx",
                        "args": [
                            "-y",
                            "tavily-mcp@0.1.2"
                        ],
                        "env": {
                            "TAVILY_API_KEY": "your-api-key-here"
                        },
                        "transport": "stdio"
                    }
                }
            }
            save_mcp_configs(default_configs)
            mcp_configs = default_configs
            logger.info(f"Created default MCP configuration file at {CONFIG_FILE}")
            
    except Exception as e:
        logger.error(f"Error loading MCP configurations: {e}")
        mcp_configs = {
            "filesystem": {
                "name": "Filesystem Server (Current Directory)",
                "description": "Access files in the current project directory",
                "config": {
                    "name": "filesystem",
                    "command": "@modelcontextprotocol/server-filesystem",
                    "args": ["."],
                    "transport": "stdio"
                }
            }
        }
    
    return mcp_configs

def save_mcp_configs(configs: Dict[str, Dict] = None) -> bool:
    """Save MCP configurations to JSON file"""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        configs_to_save = configs if configs is not None else mcp_configs
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(configs_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(configs_to_save)} MCP configurations to {CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving MCP configurations: {e}")
        return False

# Load configurations on startup
load_mcp_configs()

async def status_callback(update: StatusUpdate):
    """Callback to handle status updates from agents"""
    agent_id = update.agent_id
    if agent_id not in status_streams:
        status_streams[agent_id] = []
    status_streams[agent_id].append(update)
    
    await update_orchestration_data(agent_id, update)

async def update_orchestration_data(agent_id: str, update: StatusUpdate):
    """Update the global orchestration data and notify subscribers"""
    if agent_id not in orchestration_data:
        orchestration_data[agent_id] = {
            "agent_id": agent_id,
            "goal": "Unknown goal",
            "status": update.status.value,
            "current_activity": update.message,
            "progress": update.progress or 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "iterations_used": 0,
            "details": update.details or {},
            "recent_activities": [],
            "execution_time": 0,
            "result": ""
        }
    
    agent_data = orchestration_data[agent_id]
    agent_data["status"] = update.status.value
    agent_data["current_activity"] = update.message
    agent_data["progress"] = update.progress or agent_data["progress"]
    agent_data["last_update"] = time.time()
    agent_data["details"] = update.details or {}
    
    if update.status.value in ["completed", "failed", "cancelled"]:
        try:
            start_timestamp = agent_data["start_time"]
            end_timestamp = time.time()
            agent_data["execution_time"] = end_timestamp - start_timestamp
        except Exception as e:
            logger.error(f"Error calculating execution time: {e}")
            if update.details and "execution_time" in update.details:
                agent_data["execution_time"] = update.details["execution_time"]
            else:
                agent_data["execution_time"] = 0
    
    activity = {
        "timestamp": update.timestamp.isoformat(),
        "message": update.message,
        "progress": update.progress
    }
    agent_data["recent_activities"].append(activity)
    if len(agent_data["recent_activities"]) > 10:
        agent_data["recent_activities"] = agent_data["recent_activities"][-10:]
    
    if update.status.value == "completed" and update.details:
        logger.info(f"Processing completion details for {agent_id}: {update.details}")
        if "iterations_used" in update.details:
            agent_data["iterations_used"] = update.details["iterations_used"]
            logger.info(f"Updated iterations for {agent_id}: {update.details['iterations_used']}")
        if "result" in update.details:
            agent_data["result"] = update.details["result"]
            logger.info(f"Updated result for {agent_id}: {len(update.details['result'])} chars")
        if "final_execution_plan" in update.details:
            agent_data["execution_plan"] = update.details["final_execution_plan"]
            logger.info(f"Updated final execution plan for {agent_id}: {update.details['final_execution_plan']}")
    
    if update.details and "execution_plan" in update.details:
        new_execution_plan = update.details["execution_plan"]
        plan_updated = update.details.get("plan_updated", False)
        
        if new_execution_plan:
            agent_data["execution_plan"] = new_execution_plan
            agent_data["plan_updated"] = plan_updated
            logger.info(f"Updated execution plan for {agent_id} with {len(new_execution_plan)} steps")
        else:
            logger.info(f"No execution plan in update for {agent_id}")
    
    if update.status.value == "completed" and "Goal completed:" in update.message:
        if not agent_data.get("result"):
            result_text = update.message.replace("Goal completed: ", "").strip()
            agent_data["result"] = result_text
            logger.info(f"Extracted result from message for {agent_id}: {len(result_text)} chars")
    
    orchestration_update = {
        "type": "agent_update", 
        "agent_id": agent_id,
        "data": agent_data
    }
    
    failed_queues = []
    for queue in orchestration_subscribers:
        try:
            queue.put_nowait(orchestration_update)
        except asyncio.QueueFull:
            pass
        except Exception:
            failed_queues.append(queue)
    
    for queue in failed_queues:
        orchestration_subscribers.remove(queue)

async def cleanup_completed_agent(agent_id: str):
    """Manual cleanup function - only called when explicitly requested"""
    if agent_id in orchestration_data:
        del orchestration_data[agent_id]
        
        removal_update = {
            "type": "agent_removed",
            "agent_id": agent_id
        }
        for queue in orchestration_subscribers:
            try:
                queue.put_nowait(removal_update)
            except:
                pass

async def broadcast_orchestration_update(update_data: dict):
    """Broadcast an update to all orchestration subscribers"""
    failed_queues = []
    for queue in orchestration_subscribers:
        try:
            queue.put_nowait(update_data)
        except asyncio.QueueFull:
            failed_queues.append(queue)
        except Exception as e:
            logger.warning(f"Error broadcasting orchestration update: {e}")
            failed_queues.append(queue)
    
    # Remove failed queues
    for queue in failed_queues:
        if queue in orchestration_subscribers:
            orchestration_subscribers.remove(queue)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TELOSCRIPT - Purposeful Agent Orchestration System",
        "description": "Coordinate autonomous MCP agents toward intelligent goals",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    active_agents = orchestrator.get_active_agents()
    return {
        "status": "healthy",
        "active_agents": len(active_agents),
        "active_agent_ids": active_agents
    }

@app.post("/agents")
async def create_agent(request: Request):
    """
    Create and run an MCP agent with simplified input/output
    
    Input: Plain text goal OR JSON with goal and optional server configuration
    Output: Clean JSON with result and execution details
    
    Examples:
    
    1. Simple text goal (uses default filesystem server):
    POST /agents
    Content-Type: text/plain
    
    What does this codebase do?
    
    2. Goal with custom MCP servers:
    POST /agents
    Content-Type: application/json
    
    {
        "goal": "Search for Python async programming resources and save to a file",
        "servers": [
            {
                "name": "brave-search",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": "your-key"}
            },
            {
                "name": "filesystem", 
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
            }
        ],
        "max_iterations": 10
    }
    
    Returns:
    {
        "result": "Main answer from GPT-4.1",
        "details": "Step-by-step execution notes...",
        "execution_time": 12.34,
        "iterations_used": 3,
        "status": "completed",
        "agent_id": "uuid"
    }
    """
    try:
        # Check content type to determine how to parse the request
        content_type = request.headers.get("content-type", "").lower()
        
        if content_type.startswith("application/json"):
            # JSON request with detailed configuration
            body = await request.json()
            goal = body.get("goal", "").strip()
            servers_config = body.get("servers", [])
            max_iterations = body.get("max_iterations", 15)
            timeout = body.get("timeout", 180)
            
            if not goal:
                return {
                    "result": "Error: No goal provided in JSON request",
                    "details": "Please provide a 'goal' field in your JSON request",
                    "execution_time": 0,
                    "iterations_used": 0,
                    "status": "failed",
                    "agent_id": None
                }
            
            # Use provided servers or default to filesystem if none specified
            if not servers_config:
                servers_config = [
                    {
                        "name": "filesystem",
                        "command": "npx", 
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                        "transport": "stdio"
                    }
                ]
            
            # Convert to MCPServerConfig objects
            servers = [MCPServerConfig(**server) for server in servers_config]
            
        else:
            # Plain text request (default behavior)
            goal = await request.body()
            goal = goal.decode('utf-8').strip()
            
            if not goal:
                return {
                    "result": "Error: Empty goal provided",
                    "details": "Please provide a goal as plain text in the request body",
                    "execution_time": 0,
                    "iterations_used": 0,
                    "status": "failed",
                    "agent_id": None
                }
            
            # Default configuration with filesystem server
            servers = [
                MCPServerConfig(
                    name="filesystem",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", "."],
                    transport="stdio"
                )
            ]
            max_iterations = 15
            timeout = 180
        
        # Create agent request
        agent_id = str(uuid.uuid4())
        
        # Initialize dashboard data with goal
        orchestration_data[agent_id] = {
            "agent_id": agent_id,
            "goal": goal,
            "status": "pending",
            "current_activity": "Initializing...",
            "progress": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "iterations_used": 0,
            "details": {},
            "recent_activities": [],
            "execution_time": 0,
            "result": "",
            "selected_configs": []
        }
        
        config = MCPAgentConfig(
            servers=servers,
            goal=AgentGoal(
                description=goal,
                success_criteria="Provide comprehensive and helpful response",
                context={"launch_method": "ui", "selected_configs": []}
            ),
            max_iterations=max_iterations,
            timeout=timeout
        )
        
        agent_request = AgentRequest(id=agent_id, config=config)
        
        # Collect execution details
        execution_details = []
        
        async def detail_callback(update: StatusUpdate):
            """Collect detailed execution steps and update dashboard"""
            # Update dashboard data (this is the key fix!)
            await status_callback(update)
            
            # Also collect execution details for response
            timestamp = update.timestamp.strftime("%H:%M:%S")
            progress = f" ({update.progress:.1f}%)" if update.progress else ""
            detail_line = f"[{timestamp}] {update.message}{progress}"
            
            if update.details:
                # Format details nicely
                if isinstance(update.details, dict):
                    details_str = ", ".join([f"{k}: {v}" for k, v in update.details.items()])
                    detail_line += f" - {details_str}"
                else:
                    detail_line += f" - {update.details}"
            
            execution_details.append(detail_line)
        
        logger.info(f"Creating agent for goal: '{goal[:50]}...' with servers: {[s.name for s in servers]}")
        
        # Run the agent
        response = await orchestrator.run_single_agent(agent_request, detail_callback)
        
        logger.info(f"Orchestrator response status: {response.status} (type: {type(response.status)})")
        
        # Update dashboard with final result BEFORE returning response
        if agent_id in orchestration_data:
            orchestration_data[agent_id]["result"] = response.result
            orchestration_data[agent_id]["iterations_used"] = response.iterations_used
            orchestration_data[agent_id]["execution_time"] = response.execution_time
            orchestration_data[agent_id]["status"] = response.status.value
            orchestration_data[agent_id]["progress"] = 100.0
        
        # Format the clean response
        details_text = "\n".join(execution_details)

        api_response = {
            "result": response.result,
            "details": details_text,
            "execution_time": response.execution_time,
            "iterations_used": response.iterations_used,
            "status": response.status.value,  # Use actual response status
            "agent_id": response.agent_id,
            "configs_used": []
        }
        
        logger.info(f"API response status: {api_response['status']}")
        return api_response
        
    except Exception as e:
        logger.error(f"Agent creation error: {e}")
        return {
            "result": "Error occurred during processing",
            "details": f"Error: {str(e)}",
            "execution_time": 0,
            "iterations_used": 0,
            "status": "failed",
            "agent_id": None,
            "configs_used": []
        }

@app.post("/workflows/sequential", response_model=List[AgentResponse])
async def run_sequential_workflow(requests: List[AgentRequest]):
    """Run multiple agents sequentially"""
    try:
        logger.info(f"Running sequential workflow with {len(requests)} agents")
        
        # Initialize status streams for all agents
        for request in requests:
            status_streams[request.id] = []
            active_streams[request.id] = True
        
        workflow_id = str(uuid.uuid4())
        status_streams[workflow_id] = []
        active_streams[workflow_id] = True
        
        responses = await orchestrator.run_sequential_workflow(requests, status_callback)
        
        # Mark streams as inactive
        for request in requests:
            active_streams[request.id] = False
        active_streams[workflow_id] = False
        
        return responses
    except Exception as e:
        logger.error(f"Error running sequential workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/parallel", response_model=List[AgentResponse])
async def run_parallel_workflow(requests: List[AgentRequest]):
    """Run multiple agents in parallel"""
    try:
        logger.info(f"Running parallel workflow with {len(requests)} agents")
        
        # Initialize status streams for all agents
        for request in requests:
            status_streams[request.id] = []
            active_streams[request.id] = True
        
        workflow_id = str(uuid.uuid4())
        status_streams[workflow_id] = []
        active_streams[workflow_id] = True
        
        responses = await orchestrator.run_parallel_workflow(requests, status_callback)
        
        # Mark streams as inactive
        for request in requests:
            active_streams[request.id] = False
        active_streams[workflow_id] = False
        
        return responses
    except Exception as e:
        logger.error(f"Error running parallel workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get current status of an agent"""
    try:
        status = await orchestrator.get_agent_status(agent_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "status": status,
            "is_active": agent_id in orchestrator.get_active_agents()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/{agent_id}")
async def cancel_agent(agent_id: str):
    """Cancel a running agent"""
    try:
        success = await orchestrator.cancel_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or not running")
        
        return {"message": f"Agent {agent_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/stream")
async def stream_agent_status(agent_id: str):
    """Stream real-time status updates for an agent via Server-Sent Events"""
    
    async def event_generator():
        """Generate SSE events for status updates"""
        try:
            # Send any existing status updates
            if agent_id in status_streams:
                for update in status_streams[agent_id]:
                    yield f"data: {json.dumps(update.dict())}\n\n"
            
            # Stream new updates while agent is active
            last_update_count = len(status_streams.get(agent_id, []))
            
            while active_streams.get(agent_id, False):
                await asyncio.sleep(0.5)  # Check every 500ms
                
                if agent_id in status_streams:
                    current_updates = status_streams[agent_id]
                    new_updates = current_updates[last_update_count:]
                    
                    for update in new_updates:
                        yield f"data: {json.dumps(update.dict())}\n\n"
                    
                    last_update_count = len(current_updates)
            
            # Send final status if available
            if agent_id in status_streams and status_streams[agent_id]:
                final_update = status_streams[agent_id][-1]
                if final_update.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]:
                    yield f"data: {json.dumps(final_update.dict())}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in status stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/agents")
async def list_agents():
    """List all active agents"""
    active_agents = orchestrator.get_active_agents()
    return {
        "active_agents": active_agents,
        "count": len(active_agents)
    }

@app.get("/dashboard")
async def get_orchestration_state():
    """Get current state of all agents for orchestration interface"""
    return {
        "agents": list(orchestration_data.values()),
        "count": len(orchestration_data),
        "timestamp": time.time()
    }

@app.get("/dashboard/stream")
async def stream_orchestration():
    """Stream real-time orchestration updates for all agents via Server-Sent Events"""
    
    async def orchestration_event_generator():
        """Generate SSE events for orchestration updates"""
        # Create a queue for this subscriber
        subscriber_queue = asyncio.Queue(maxsize=100)
        orchestration_subscribers.append(subscriber_queue)
        
        try:
            # Send initial state
            initial_state = {
                "type": "initial_state",
                "agents": list(orchestration_data.values()),
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(initial_state)}\n\n"
            
            # Stream updates
            while True:
                try:
                    # Wait for update with timeout
                    update = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    ping = {
                        "type": "ping",
                        "timestamp": time.time()
                    }
                    yield f"data: {json.dumps(ping)}\n\n"
                except Exception as e:
                    logger.error(f"Orchestration stream error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Orchestration stream setup error: {e}")
        finally:
            # Clean up subscriber
            if subscriber_queue in orchestration_subscribers:
                orchestration_subscribers.remove(subscriber_queue)
    
    return StreamingResponse(
        orchestration_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.delete("/agents")
async def cancel_all_agents():
    """Cancel all running agents"""
    try:
        cancelled_count = await orchestrator.cancel_all_agents()
        return {
            "message": f"Cancelled {cancelled_count} agents",
            "cancelled_count": cancelled_count
        }
    except Exception as e:
        logger.error(f"Error cancelling all agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples/mcp-config")
async def get_mcp_config_example():
    """Get example MCP configurations for different use cases"""
    return {
        "filesystem_analysis": {
            "goal": "Analyze the structure and purpose of this codebase",
            "servers": [
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                    "transport": "stdio"
                }
            ],
            "max_iterations": 12,
            "timeout": 300
        },
        "web_research_and_save": {
            "goal": "Research the latest developments in AI agents and save findings to a report",
            "servers": [
                {
                    "name": "brave-search",
                    "command": "npx", 
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "your-brave-api-key"},
                    "transport": "stdio"
                },
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "./reports"],
                    "transport": "stdio"
                }
            ],
            "max_iterations": 15,
            "timeout": 400
        },
        "github_analysis": {
            "goal": "Analyze recent commits and issues in a GitHub repository",
            "servers": [
                {
                    "name": "github",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"},
                    "transport": "stdio"
                }
            ],
            "max_iterations": 10,
            "timeout": 250
        },
        "multi_server_workflow": {
            "goal": "Research a topic online, analyze related code files, and generate a comprehensive report",
            "servers": [
                {
                    "name": "brave-search",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "your-api-key"},
                    "transport": "stdio"
                },
                {
                    "name": "filesystem",
                    "command": "npx", 
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                    "transport": "stdio"
                },
                {
                    "name": "github",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"},
                    "transport": "stdio"
                }
            ],
            "max_iterations": 20,
            "timeout": 600
        },
        "database_query": {
            "goal": "Query database for user analytics and generate insights",
            "servers": [
                {
                    "name": "database",
                    "command": "python",
                    "args": ["-m", "custom_database_server"],
                    "env": {"DB_CONNECTION_STRING": "your-connection-string"},
                    "transport": "stdio"
                }
            ],
            "max_iterations": 8,
            "timeout": 180
        }
    }

@app.get("/examples/request")
async def get_request_example():
    """Get example agent request"""
    return {
        "id": "example-agent-123",
        "config": {
            "servers": [
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "transport": "stdio"
                }
            ],
            "goal": {
                "description": "List files in the current directory and create a summary",
                "success_criteria": "file list created",
                "priority": 1
            },
            "max_iterations": 5,
            "timeout": 120
        }
    }

@app.get("/dashboard/test")
async def orchestration_interface():
    """TELOSCRIPT Orchestration Interface - Main web interface for purposeful agent coordination"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 TELOSCRIPT - Purposeful Agent Orchestration</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            /* Purposeful & Goal-Directed Colors */
            --primary-gradient: linear-gradient(135deg, #1a1c29 0%, #2d1b69 25%, #4a148c 50%, #6a1b9a 75%, #8e24aa 100%);
            --orchestration-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            --emergence-gradient: linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%);
            --autonomous-gradient: linear-gradient(135deg, #dc2626 0%, #ea580c 50%, #f59e0b 100%);
            
            /* Philosophical Accent Colors */
            --wisdom-gold: #fbbf24;
            --purpose-purple: #8b5cf6;
            --intelligence-cyan: #06b6d4;
            --harmony-green: #10b981;
            
            /* Glass Morphism with Purpose */
            --glass-primary: rgba(139, 92, 246, 0.08);
            --glass-secondary: rgba(6, 182, 212, 0.06);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.25);
            
            /* Typography for Intelligence */
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: rgba(248, 250, 252, 0.6);
            --text-accent: var(--wisdom-gold);
        }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--primary-gradient);
            background-attachment: fixed;
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
            position: relative;
        }
        
        /* Orchestration Grid Pattern */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 25% 25%, rgba(139, 92, 246, 0.1) 0%, transparent 25%),
                radial-gradient(circle at 75% 75%, rgba(6, 182, 212, 0.08) 0%, transparent 25%),
                linear-gradient(45deg, transparent 40%, rgba(251, 191, 36, 0.02) 50%, transparent 60%);
            animation: orchestrate 30s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes orchestrate {
            0%, 100% { transform: translateY(0px) rotate(0deg) scale(1); opacity: 0.4; }
            33% { transform: translateY(-10px) rotate(1deg) scale(1.02); opacity: 0.6; }
            66% { transform: translateY(5px) rotate(-0.5deg) scale(0.98); opacity: 0.3; }
        }
        
        .header {
            background: var(--glass-primary);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            padding: 30px 20px;
            box-shadow: var(--glass-shadow);
            position: sticky;
            top: 0;
            z-index: 100;
            border-radius: 0 0 24px 24px;
            margin-bottom: 30px;
        }
        
        .brand-container {
            max-width: 1400px;
            margin: 0 auto;
            text-align: center;
            position: relative;
        }
        
        /* Philosophical Quote */
        .philosophy-quote {
            font-style: italic;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 15px;
            font-weight: 300;
            letter-spacing: 0.3px;
        }
        
        .philosophy-quote::before,
        .philosophy-quote::after {
            content: '"';
            color: var(--wisdom-gold);
            font-size: 1.2em;
        }
        
        .header h1 {
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 800;
            text-align: center;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            letter-spacing: -0.02em;
            position: relative;
        }
        
        .header h1 .emoji {
            font-size: 0.9em;
            filter: drop-shadow(0 4px 12px rgba(139, 92, 246, 0.4));
            animation: purposefulPulse 4s ease-in-out infinite;
        }
        
        @keyframes purposefulPulse {
            0%, 100% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.05) rotate(1deg); }
            50% { transform: scale(1.08) rotate(0deg); }
            75% { transform: scale(1.03) rotate(-1deg); }
        }
        
        .header h1 .title-text {
            background: linear-gradient(135deg, var(--wisdom-gold) 0%, var(--purpose-purple) 40%, var(--intelligence-cyan) 80%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
        }
        
        .header h1 .title-text::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--wisdom-gold), transparent);
            animation: purposeFlow 3s ease-in-out infinite;
        }
        
        @keyframes purposeFlow {
            0%, 100% { transform: scaleX(0.3); opacity: 0.3; }
            50% { transform: scaleX(1); opacity: 1; }
        }
        
        .header .subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-style: italic;
            font-size: clamp(1rem, 2.5vw, 1.4rem);
            font-weight: 400;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        
        .subtitle-accent {
            color: var(--wisdom-gold);
            font-weight: 600;
            font-style: normal;
        }
        
        .version-badge {
            display: inline-block;
            background: var(--emergence-gradient);
            color: white;
            padding: 6px 16px;
            border-radius: 24px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 12px;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .orchestration-status {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 30px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--harmony-green);
            animation: harmonicPulse 2s ease-in-out infinite;
        }
        
        @keyframes harmonicPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
        }
        
        .nav-tabs {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 25px;
        }
        
        .nav-tab {
            padding: 14px 28px;
            background: var(--glass-secondary);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 600;
            color: var(--text-secondary);
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
        }
        
        .nav-tab::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.2), transparent);
            transition: left 0.6s ease;
        }
        
        .nav-tab:hover::before {
            left: 100%;
        }
        
        .nav-tab.active {
            background: var(--glass-primary);
            border-color: var(--purpose-purple);
            color: var(--text-primary);
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);
            transform: translateY(-2px);
        }
        
        .nav-tab:hover:not(.active) {
            background: rgba(139, 92, 246, 0.1);
            transform: translateY(-1px);
            color: var(--text-primary);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px 40px;
            position: relative;
            z-index: 1;
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .orchestration-grid {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .agents-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        
        .section {
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--glass-shadow);
            margin-bottom: 25px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: var(--secondary-gradient);
            opacity: 0.6;
        }
        
        .section:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 60px 0 rgba(31, 38, 135, 0.5);
        }
        
        .section h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 25px;
            color: var(--text-light);
            border-bottom: 2px solid rgba(255,255,255,0.1);
            padding-bottom: 15px;
            letter-spacing: -0.01em;
        }
        
        .launcher-section {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
            border-left: 4px solid;
            border-image: var(--secondary-gradient) 1;
        }
        
        .launcher-section::before {
            background: var(--secondary-gradient);
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .config-card {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--glass-shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .config-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-gradient);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .config-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .config-card:hover::before {
            opacity: 1;
        }
        
        .config-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .config-name {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text-light);
            letter-spacing: -0.01em;
        }
        
        .config-description {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 18px;
            line-height: 1.5;
        }
        
        .config-details {
            background: rgba(0,0,0,0.1);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'Monaco', 'Menlo', monospace;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .btn {
            background: var(--secondary-gradient);
            color: var(--text-primary);
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin: 6px;
            width: 100%;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.2);
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.6s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }

        .agent-card .btn {
            color: black;
        }
        
        .btn-secondary {
            background: var(--dark-gradient);
            box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
        }
        
        .btn-secondary:hover {
            box-shadow: 0 8px 25px rgba(44, 62, 80, 0.5);
        }
        
        .btn-danger {
            background: var(--warning-gradient);
            box-shadow: 0 4px 15px rgba(252, 74, 26, 0.3);
        }
        
        .btn-danger:hover {
            box-shadow: 0 8px 25px rgba(252, 74, 26, 0.5);
        }
        
        .btn-small {
            padding: 10px 20px;
            font-size: 0.85rem;
            width: auto;
        }
        
        .form-group {
            margin-bottom: 24px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-light);
            font-size: 0.9rem;
            letter-spacing: 0.3px;
        }
        
        .form-input, .form-textarea {
            width: 100%;
            padding: 14px 16px;
            border: 1px solid var(--glass-border);
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            font-size: 0.9rem;
            color: var(--text-light);
            transition: all 0.3s ease;
            font-family: inherit;
        }
        
        .form-input::placeholder, .form-textarea::placeholder {
            color: var(--text-muted);
        }
        
        .form-input:focus, .form-textarea:focus {
            outline: none;
            border-color: #4facfe;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
            background: rgba(255,255,255,0.05);
        }
        
        .form-textarea {
            min-height: 120px;
            font-family: 'Monaco', 'Menlo', monospace;
            resize: vertical;
            line-height: 1.5;
        }
        
        .config-selector {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 20px;
        }

        /* Purpose Endpoints Styles */
        .purpose-endpoints-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .purpose-endpoint-card {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--glass-shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .purpose-endpoint-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(135deg, var(--wisdom-gold) 0%, var(--purpose-purple) 50%, var(--intelligence-cyan) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .purpose-endpoint-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 20px 40px rgba(139, 92, 246, 0.2);
        }
        
        .purpose-endpoint-card:hover::before {
            opacity: 1;
        }
        
        .endpoint-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }
        
        .endpoint-name {
            font-weight: 700;
            font-size: 1.2rem;
            color: var(--text-primary);
            letter-spacing: -0.01em;
            margin-bottom: 6px;
        }
        
        .endpoint-slug-container {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .endpoint-slug {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--purpose-purple);
            background: rgba(139, 92, 246, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        .btn-copy-url {
            background: rgba(139, 92, 246, 0.08);
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 4px;
            padding: 2px 6px;
            cursor: pointer;
            font-size: 0.75rem;
            color: var(--purpose-purple);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 24px;
            height: 20px;
            opacity: 0.7;
        }
        
        .btn-copy-url:hover {
            background: rgba(139, 92, 246, 0.15);
            border-color: rgba(139, 92, 246, 0.3);
            opacity: 1;
            transform: scale(1.05);
        }
        
        .btn-copy-url:active {
            transform: scale(0.95);
            background: rgba(139, 92, 246, 0.2);
        }
        
        .endpoint-description {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 18px;
            line-height: 1.5;
        }
        
        .endpoint-meta {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 20px;
        }
        
        .endpoint-servers {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .server-tag {
            background: rgba(6, 182, 212, 0.1);
            color: var(--intelligence-cyan);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            border: 1px solid rgba(6, 182, 212, 0.2);
        }
        
        .endpoint-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .endpoint-tag {
            background: rgba(251, 191, 36, 0.1);
            color: var(--wisdom-gold);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 500;
            border: 1px solid rgba(251, 191, 36, 0.2);
        }
        
        .endpoint-status {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }
        
        .endpoint-status.enabled {
            color: var(--harmony-green);
        }
        
        .endpoint-status.disabled {
            color: var(--text-muted);
        }
        
        .endpoint-actions {
            display: flex;
            gap: 8px;
            margin-top: 16px;
            align-items: center;
        }
        
        .btn-test {
            background: var(--emergence-gradient);
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        
        .btn-edit {
            background: var(--orchestration-gradient);
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        
        .btn-delete {
            background: var(--autonomous-gradient);
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        
        .config-checkbox {
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
            background: var(--glass-secondary);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .config-checkbox:hover {
            background: rgba(139, 92, 246, 0.1);
            border-color: var(--purpose-purple);
        }
        
        .config-checkbox input {
            margin-right: 10px;
            transform: scale(1.2);
            accent-color: var(--purpose-purple);
        }
        
        .config-checkbox-label {
            flex: 1;
        }
        
        .config-checkbox-name {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .config-checkbox-desc {
            font-size: 0.8em;
            color: var(--text-secondary);
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
        }
        
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--glass-primary);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
            color: var(--text-primary);
            padding: 30px;
            border-radius: 15px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--glass-border);
        }
        
        .modal-header h3 {
            color: var(--text-primary);
            margin: 0;
        }
        
        .close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--text-muted);
            transition: color 0.2s ease;
        }
        
        .close:hover {
            color: var(--text-primary);
        }
        
        .agent-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 4px solid #cbd5e0;
            transition: all 0.3s ease;
            color: #2d3748; /* Ensure text is dark and visible */
        }
        
        .agent-card.running { border-left-color: #4ade80; }
        .agent-card.completed { border-left-color: #3b82f6; }
        .agent-card.failed { border-left-color: #f87171; }
        
        .empty-state {
            text-align: center;
            color: #a0aec0;
            font-style: italic;
            padding: 40px 20px;
        }
        
        .advanced-settings {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .terminal-container {
            background: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
            font-family: 'Courier New', 'Monaco', monospace;
            border: 1px solid #333;
        }
        
        .terminal-header {
            background: #2a2a2a;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }
        
        .terminal-title {
            color: #00ff00;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .terminal-clear {
            background: #444;
            color: #ccc;
            border: 1px solid #555;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            cursor: pointer;
        }
        
        .terminal-clear:hover {
            background: #555;
        }
        
        .terminal-output {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            background: #000;
            color: #00ff00;
            font-size: 0.85em;
            line-height: 1.4;
        }
        
        .terminal-line {
            margin-bottom: 2px;
            word-wrap: break-word;
        }
        
        .terminal-timestamp {
            color: #888;
        }
        
        .terminal-text {
            color: #00ff00;
        }
        
        .terminal-error {
            color: #ff6b6b;
        }
        
        .terminal-success {
            color: #51cf66;
        }
        
        .terminal-info {
            color: #74c0fc;
        }
        
        .agent-card .current-activity {
            background: #f0f8ff;
            border-left: 3px solid #4dabf7;
            padding: 8px 12px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.9em;
            animation: pulse-blue 2s infinite;
        }
        
        @keyframes pulse-blue {
            0% { border-left-color: #4dabf7; }
            50% { border-left-color: #228be6; }
            100% { border-left-color: #4dabf7; }
        }
        
        .agent-card .recent-activities {
            margin-top: 10px;
            max-height: 100px;
            overflow-y: auto;
        }
        
        .activity-item {
            font-size: 0.8em;
            color: #666;
            padding: 3px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .activity-timestamp {
            color: #999;
            font-size: 0.75em;
        }
        
        .details-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1001;
        }
        
        .details-modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
            color: var(--text-primary);
            padding: 30px;
            border-radius: 15px;
            max-width: 900px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .details-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--glass-border);
        }
        
        .details-tab {
            padding: 10px 15px;
            background: var(--glass-secondary);
            border: 1px solid var(--glass-border);
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-weight: 600;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }
        
        .details-tab.active {
            background: var(--purpose-purple);
            color: var(--text-primary);
            border-color: var(--purpose-purple);
        }
        
        .details-content {
            display: none;
        }
        
        .details-content.active {
            display: block;
        }
        
        .execution-log {
            background: #1a1a1a;
            color: #00ff00;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        @media (max-width: 1200px) {
            .orchestration-grid {
                grid-template-columns: 1fr;
            }
            .agents-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .nav-tabs {
                flex-wrap: wrap;
            }
            .advanced-settings {
                grid-template-columns: 1fr;
            }
        }
        
        .execution-plan {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            font-size: 0.9em;
            color: #2d3748; /* Ensure text is dark */
        }
        
        .execution-plan-title {
            font-weight: 600;
            color: #2d3748; /* Ensure title is dark */
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .execution-plan-steps {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .execution-plan-step {
            padding: 4px 0;
            line-height: 1.4;
            display: flex;
            align-items: flex-start;
            gap: 8px;
            color: #2d3748; /* Ensure step text is dark */
        }
        
        .execution-plan-step .step-indicator {
            font-size: 1.1em;
            min-width: 20px;
        }
        
        .execution-plan-step .step-text {
            flex: 1;
            color: #4a5568; /* Ensure step text is dark grey */
        }
        
        .plan-updated-badge {
            background: #3b82f6;
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
        }
        
        .details-modal-content {
            padding: 30px;
            border-radius: 15px;
            max-width: 900px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            color: #2d3748; /* Ensure all modal text is dark */
        }
        
        .details-modal-content h4 {
            color: #2d3748; /* Ensure headings are dark */
        }
        
        .details-modal-content p {
            color: #4a5568; /* Ensure paragraphs are dark grey */
        }
        
        /* Toast Notification System */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        }
        
        .toast {
            background: var(--glass-primary);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: var(--glass-shadow);
            max-width: 400px;
            min-width: 320px;
            pointer-events: all;
            transform: translateX(100%);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .toast.show {
            transform: translateX(0);
            opacity: 1;
        }
        
        .toast.success {
            border-left: 4px solid var(--harmony-green);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), var(--glass-primary));
        }
        
        .toast.error {
            border-left: 4px solid #ef4444;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), var(--glass-primary));
        }
        
        .toast.info {
            border-left: 4px solid var(--intelligence-cyan);
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), var(--glass-primary));
        }
        
        .toast::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--wisdom-gold), transparent);
            animation: toastShimmer 2s ease-in-out infinite;
        }
        
        @keyframes toastShimmer {
            0%, 100% { transform: scaleX(0.3); opacity: 0.3; }
            50% { transform: scaleX(1); opacity: 1; }
        }
        
        .toast-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        
        .toast-icon {
            font-size: 1.5em;
            flex-shrink: 0;
            animation: toastIconPulse 1.5s ease-in-out infinite;
        }
        
        @keyframes toastIconPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .toast-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        
        .toast-close {
            margin-left: auto;
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 1.2em;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        .toast-close:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }
        
        .toast-message {
            color: var(--text-secondary);
            line-height: 1.5;
            font-size: 0.95em;
        }
        
        .toast-details {
            margin-top: 12px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .toast-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 8px;
            font-size: 0.85em;
        }
        
        .toast-metric {
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
        }
        
        .toast-metric-value {
            color: var(--wisdom-gold);
            font-weight: 600;
        }
        
        /* Progress Bar for Toast */
        .toast-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--purpose-purple), var(--intelligence-cyan));
            border-radius: 0 0 16px 16px;
            transform-origin: left;
            animation: toastProgress 5s linear forwards;
        }
        
        @keyframes toastProgress {
            from { width: 100%; }
            to { width: 0%; }
        }
        
        /* Celebration Animation for Success */
        .toast.success .toast-icon {
            animation: celebrationBounce 0.6s ease-out;
        }
        
        @keyframes celebrationBounce {
            0% { transform: scale(0.3) rotate(-10deg); }
            50% { transform: scale(1.3) rotate(10deg); }
            100% { transform: scale(1) rotate(0deg); }
        }
        
        /* Audio Controls (Hidden) */
        .audio-controls {
            position: absolute;
            left: -9999px;
            opacity: 0;
        }
    </style>
</head>
<body>
    <!-- Toast Notification Container -->
    <div class="toast-container" id="toast-container"></div>
    
    <!-- Audio Elements for Notifications -->
    <div class="audio-controls">
        <audio id="success-sound" preload="auto"></audio>
        <audio id="error-sound" preload="auto"></audio>
        <audio id="info-sound" preload="auto"></audio>
    </div>

    <div class="header">
        <div class="brand-container">
            <div class="philosophy-quote">He who has a why to live can bear almost any how — Friedrich Nietzsche</div>
            <h1><span class="emoji">🎯</span><span class="title-text">TELOSCRIPT</span></h1>
            <div class="subtitle">
                <span class="subtitle-accent">Purposeful</span> Agent Orchestration System
                <div class="version-badge">v1.1.0 Purposeful</div>
            </div>
            <div class="orchestration-status">
                <div class="status-item">
                    <div class="status-dot"></div>
                    <span>Autonomous Agents</span>
                </div>
                <div class="status-item">
                    <div class="status-dot"></div>
                    <span>Emergent Intelligence</span>
                </div>
                <div class="status-item">
                    <div class="status-dot"></div>
                    <span>Goal-Directed</span>
                </div>
            </div>
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="showTab('orchestration')">🎼 Orchestration</div>
                <div class="nav-tab" onclick="showTab('configs')">🔧 MCP Configuration</div>
                <div class="nav-tab" onclick="showTab('purpose-endpoints')">🎯 Purpose Endpoints</div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <!-- Orchestration Tab -->
        <div id="orchestration" class="tab-content active">
            <div class="orchestration-grid">
                <!-- Agent Launcher Panel -->
                <div class="section launcher-section">
                    <h2>🎯 Define Purpose & Launch Agent</h2>
                    
                    <div class="form-group">
                        <label class="form-label">Agent's Purpose (Goal)</label>
                        <textarea id="launch-goal" class="form-textarea" placeholder="Define what meaningful objective you want the agent to accomplish with autonomous purpose..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Agent Capabilities (MCP Servers)</label>
                        <div class="config-selector" id="config-selector">
                            <div class="empty-state">Loading available capabilities...</div>
                        </div>
                    </div>
                    
                    <div class="advanced-settings">
                        <div class="form-group">
                            <label class="form-label">Iteration Limit</label>
                            <input type="number" id="launch-iterations" class="form-input" value="15" min="1" max="50">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Max Time (seconds)</label>
                            <input type="number" id="launch-timeout" class="form-input" value="180" min="30" max="600">
                        </div>
                    </div>
                    
                    <button class="btn" onclick="launchAgent()">🚀 Orchestrate Agent</button>
                </div>
                
                <!-- Live Activity Terminal -->
                <div class="section">
                    <h2>🎼 Agent Orchestration Monitor</h2>
                    <div class="terminal-container">
                        <div class="terminal-header">
                            <span class="terminal-title">🟢 TELOSCRIPT Autonomous Activity</span>
                            <button class="terminal-clear" onclick="clearTerminal()">Clear</button>
                        </div>
                        <div class="terminal-output" id="terminal-output">
                            <div class="terminal-line">
                                <span class="terminal-timestamp" id="initial-timestamp"></span>
                                <span class="terminal-text">TELOSCRIPT ready - Awaiting purposeful agent coordination...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="agents-grid">
                <div class="section">
                    <h2>⚡ Active Orchestration</h2>
                    <div id="active-agents">
                        <div class="empty-state">No agents currently orchestrating</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>✨ Completed Purposes</h2>
                    <div id="completed-agents">
                        <div class="empty-state">No purposes yet fulfilled</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- MCP Configs Tab -->
        <div id="configs" class="tab-content">
            <div class="section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div>
                        <h2>🔧 MCP Server Configurations</h2>
                        <p style="color: var(--text-muted); font-size: 0.9em; margin: 5px 0;">
                            Define the tools and capabilities available to autonomous agents
                        </p>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-secondary btn-small" onclick="reloadConfigs()">🔄 Reload</button>
                        <button class="btn btn-small" onclick="showCreateConfigModal()">➕ New Config</button>
                    </div>
                </div>
                <div class="config-grid" id="config-grid">
                    <div class="empty-state">Loading configurations...</div>
                </div>
            </div>
        </div>

        <!-- Purpose Endpoints Tab -->
        <div id="purpose-endpoints" class="tab-content">
            <div class="section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div>
                        <h2>🎯 Purpose Endpoints</h2>
                        <p style="color: var(--text-muted); font-size: 0.9em; margin: 5px 0;">
                            Predefined agent configurations with baked-in prompts and MCP servers
                        </p>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-secondary btn-small" onclick="refreshPurposeEndpoints()">🔄 Refresh</button>
                        <button class="btn btn-small" onclick="showCreateEndpointModal()">➕ New Endpoint</button>
                    </div>
                </div>
                <div class="purpose-endpoints-grid" id="purpose-endpoints-grid">
                    <div class="empty-state">Loading purpose endpoints...</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Config Creation/Edit Modal -->
    <div id="config-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-title">Create MCP Configuration</h3>
                <button class="close" onclick="closeConfigModal()">&times;</button>
            </div>
            <form id="config-form">
                <div class="form-group">
                    <label class="form-label">Configuration ID</label>
                    <input type="text" id="config-id" class="form-input" placeholder="e.g., my-file-server" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Display Name</label>
                    <input type="text" id="config-name" class="form-input" placeholder="e.g., My File Server" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" id="config-description" class="form-input" placeholder="Brief description of what this config does" required>
                </div>
                <div class="form-group">
                    <label class="form-label">MCP Server Configuration (JSON)</label>
                    <textarea id="config-json" class="form-textarea" placeholder='{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."], "transport": "stdio"}' required></textarea>
                </div>
                <div style="display: flex; gap: 10px; justify-content: flex-end;">
                    <button type="button" class="btn btn-secondary btn-small" onclick="closeConfigModal()">Cancel</button>
                    <button type="submit" class="btn btn-small">Save Configuration</button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- Agent Details Modal -->
    <div id="details-modal" class="details-modal">
        <div class="details-modal-content">
            <div class="modal-header">
                <h3 id="details-title">Agent Details</h3>
                <button class="close" onclick="closeDetailsModal()">&times;</button>
            </div>
            <div class="details-tabs">
                <button class="details-tab active" onclick="showDetailsTab('overview')">Overview</button>
                <button class="details-tab" onclick="showDetailsTab('plan')">Execution Plan</button>
                <button class="details-tab" onclick="showDetailsTab('execution')">Execution Log</button>
                <button class="details-tab" onclick="showDetailsTab('result')">Result</button>
                <button class="details-tab" onclick="showDetailsTab('config')">Configuration</button>
            </div>
            <div id="details-overview" class="details-content active">
                <div id="details-overview-content"></div>
            </div>
            <div id="details-plan" class="details-content">
                <div id="details-plan-content"></div>
            </div>
            <div id="details-execution" class="details-content">
                <div class="execution-log" id="details-execution-log"></div>
            </div>
            <div id="details-result" class="details-content">
                <div id="details-result-content"></div>
            </div>
            <div id="details-config" class="details-content">
                <div id="details-config-content"></div>
            </div>
        </div>
    </div>

    <!-- Purpose Endpoint Creation/Edit Modal -->
    <div id="purpose-endpoint-modal" class="modal">
        <div class="modal-content" style="max-width: 800px;">
            <div class="modal-header">
                <h3 id="purpose-modal-title">Create Purpose Endpoint</h3>
                <button class="close" onclick="closePurposeEndpointModal()">&times;</button>
            </div>
            <form id="purpose-endpoint-form">
                <div class="form-group">
                    <label class="form-label">Endpoint Slug</label>
                    <input type="text" id="purpose-slug" class="form-input" placeholder="e.g., handle-webhook" required>
                    <small style="color: var(--text-muted); font-size: 0.8em;">URL-friendly identifier (lowercase, hyphens only)</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Display Name</label>
                    <input type="text" id="purpose-name" class="form-input" placeholder="e.g., GitHub Webhook Handler" required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" id="purpose-description" class="form-input" placeholder="Brief description of what this endpoint does" required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Prompt Template</label>
                    <textarea id="purpose-prompt" class="form-textarea" style="min-height: 150px;" placeholder="Enter the agent's purpose and instructions. The input data will be automatically appended to your prompt." required></textarea>
                    <small style="color: var(--text-muted); font-size: 0.8em;">Input data will be automatically appended to your prompt when the endpoint is executed</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">MCP Servers</label>
                    <div id="purpose-mcp-selector" class="config-selector">
                        <div class="empty-state">Loading available MCP servers...</div>
                    </div>
                    <small style="color: var(--text-muted); font-size: 0.8em;">Select the MCP servers this endpoint should have access to</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Tags (comma-separated)</label>
                    <input type="text" id="purpose-tags" class="form-input" placeholder="e.g., webhook, github, automation">
                    <small style="color: var(--text-muted); font-size: 0.8em;">Optional tags for organization and filtering</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">
                        <input type="checkbox" id="purpose-enabled" checked style="margin-right: 8px;">
                        Enabled
                    </label>
                    <small style="color: var(--text-muted); font-size: 0.8em; display: block; margin-top: 4px;">Whether this endpoint is active and can be executed</small>
                </div>
                
                <div style="display: flex; gap: 10px; justify-content: flex-end; margin-top: 30px;">
                    <button type="button" class="btn btn-secondary btn-small" onclick="closePurposeEndpointModal()">Cancel</button>
                    <button type="submit" class="btn btn-small">Save Purpose Endpoint</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        let configs = {};
        let agents = {};
        let editingConfigId = null;
        
        // Toast Notification System
        function showToast(type, title, message, details = null) {
            const container = document.getElementById('toast-container');
            const toastId = 'toast-' + Date.now();
            
            // Choose icon based on type
            const icons = {
                success: '🎉',
                error: '❌', 
                info: 'ℹ️'
            };
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.id = toastId;
            
            toast.innerHTML = `
                <div class="toast-header">
                    <div class="toast-icon">${icons[type] || 'ℹ️'}</div>
                    <div class="toast-title">${title}</div>
                    <button class="toast-close" onclick="removeToast('${toastId}')">&times;</button>
                </div>
                <div class="toast-message">${message}</div>
                ${details ? `<div class="toast-details">${details}</div>` : ''}
                <div class="toast-progress"></div>
            `;
            
            container.appendChild(toast);
            
            // Trigger show animation
            setTimeout(() => toast.classList.add('show'), 100);
            
            // Play sound
            playNotificationSound(type);
            
            // Auto-remove after 5 seconds
            setTimeout(() => removeToast(toastId), 5000);
            
            return toastId;
        }
        
        function removeToast(toastId) {
            const toast = document.getElementById(toastId);
            if (toast) {
                toast.classList.remove('show');
                setTimeout(() => {
                    if (toast.parentNode) {
                        toast.parentNode.removeChild(toast);
                    }
                }, 400);
            }
        }
        
        function playNotificationSound(type) {
            try {
                // Create Web Audio context for synthesized sounds
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                // Sound parameters based on notification type
                const soundConfigs = {
                    success: {
                        frequencies: [523.25, 659.25, 783.99], // C5, E5, G5 - Happy chord
                        duration: 0.3,
                        type: 'sine'
                    },
                    error: {
                        frequencies: [349.23, 329.63], // F4, E4 - Dissonant
                        duration: 0.4,
                        type: 'sawtooth'
                    },
                    info: {
                        frequencies: [440, 554.37], // A4, C#5 - Neutral pleasant
                        duration: 0.2,
                        type: 'sine'
                    }
                };
                
                const config = soundConfigs[type] || soundConfigs.info;
                
                // Create and play tones
                config.frequencies.forEach((freq, index) => {
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.value = freq;
                    oscillator.type = config.type;
                    
                    // Volume envelope
                    const now = audioContext.currentTime;
                    const delay = index * 0.1; // Stagger notes
                    gainNode.gain.setValueAtTime(0, now + delay);
                    gainNode.gain.linearRampToValueAtTime(0.1, now + delay + 0.05);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, now + delay + config.duration);
                    
                    oscillator.start(now + delay);
                    oscillator.stop(now + delay + config.duration);
                });
                
            } catch (e) {
                // Fallback to simple beep if Web Audio API is not available
                console.log('Web Audio API not available, using fallback:', e);
                try {
                    const audio = document.getElementById(`${type}-sound`);
                    if (audio) {
                        // Create a simple data URI beep sound as fallback
                        const beepSound = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1fLMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEbBTeR1fPRdSUGI3fB8eFOQggUVq/n569VFApFpeDyx2EaDD+Y2e7JciEGH3O58N7jWCwGJG3L8dZPNQwEUrPl8aiEPgsHY5nN8JqrRAQCGHCO5AWBR2E";
                        audio.src = beepSound;
                        audio.currentTime = 0;
                        audio.play().catch(() => {});
                    }
                } catch (fallbackError) {
                    // Silent fallback
                    console.log('Audio completely unavailable');
                }
            }
        }
        
        function showAgentCompletionToast(result) {
            if (result.status === 'completed') {
                const detailsHtml = `
                    <div class="toast-metrics">
                        <div class="toast-metric">
                            <span>Execution Time:</span>
                            <span class="toast-metric-value">${result.execution_time.toFixed(2)}s</span>
                        </div>
                        <div class="toast-metric">
                            <span>Iterations:</span>
                            <span class="toast-metric-value">${result.iterations_used}</span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em; max-height: 80px; overflow-y: auto;">
                        <strong>Result Preview:</strong><br>
                        ${result.result.substring(0, 150)}${result.result.length > 150 ? '...' : ''}
                    </div>
                `;
                
                showToast('success', '🎯 Agent Completed Successfully!', 
                    'Your autonomous agent has fulfilled its purpose with intelligence and efficiency.', 
                    detailsHtml);
                
                // Add celebration effect
                triggerCelebrationEffect();
                
            } else {
                showToast('error', '❌ Agent Failed', 
                    'The agent encountered an issue and could not complete its task.', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${result.result}</div>`);
            }
        }
        
        function triggerCelebrationEffect() {
            // Create floating celebration emojis
            const emojis = ['🎉', '✨', '🎯', '🚀', '⭐', '💫'];
            for (let i = 0; i < 6; i++) {
                setTimeout(() => {
                    const emoji = document.createElement('div');
                    emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];
                    emoji.style.cssText = `
                        position: fixed;
                        top: 20%;
                        left: ${Math.random() * 100}%;
                        font-size: 2rem;
                        pointer-events: none;
                        z-index: 3000;
                        animation: celebrate 3s ease-out forwards;
                    `;
                    
                    document.body.appendChild(emoji);
                    setTimeout(() => emoji.remove(), 3000);
                }, i * 200);
            }
            
            // Add CSS animation for celebration
            if (!document.getElementById('celebration-styles')) {
                const styles = document.createElement('style');
                styles.id = 'celebration-styles';
                styles.textContent = `
                    @keyframes celebrate {
                        0% { transform: translateY(0) rotate(0deg) scale(0.5); opacity: 1; }
                        50% { transform: translateY(-100px) rotate(180deg) scale(1.2); opacity: 0.8; }
                        100% { transform: translateY(-200px) rotate(360deg) scale(0.3); opacity: 0; }
                    }
                `;
                document.head.appendChild(styles);
            }
        }
        
        // Keyboard support for dismissing toasts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                // Dismiss all toasts
                const toasts = document.querySelectorAll('.toast.show');
                toasts.forEach(toast => {
                    removeToast(toast.id);
                });
            }
        });
        
        // Replace alert functions with toast notifications
        function showSuccessNotification(title, message, details = null) {
            showToast('success', title, message, details);
        }
        
        function showErrorNotification(title, message, details = null) {
            showToast('error', title, message, details);
        }
        
        function showInfoNotification(title, message, details = null) {
            showToast('info', title, message, details);
        }

        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'configs') {
                loadConfigs();
            } else if (tabName === 'orchestration') {
                loadConfigSelector();
            } else if (tabName === 'purpose-endpoints') {
                loadPurposeEndpoints();
            }
        }
        
        // Load MCP configurations
        async function loadConfigs() {
            try {
                const response = await fetch('/mcp-configs');
                const data = await response.json();
                configs = data.configs;
                renderConfigs();
            } catch (error) {
                console.error('Error loading configs:', error);
            }
        }
        
        // Render configuration cards
        function renderConfigs() {
            const grid = document.getElementById('config-grid');
            if (Object.keys(configs).length === 0) {
                grid.innerHTML = '<div class="empty-state">No configurations found. Create your first one!</div>';
                return;
            }
            
            grid.innerHTML = Object.entries(configs).map(([id, config]) => `
                <div class="config-card">
                    <div class="config-header">
                        <div class="config-name">${config.name}</div>
                    </div>
                    <div class="config-description">${config.description}</div>
                    <div class="config-details">
                        <strong>Command:</strong> ${config.config.command}<br>
                        <strong>Server:</strong> ${config.config.name}
                    </div>
                    <div style="margin-top: 15px; display: flex; gap: 10px;">
                        <button class="btn btn-secondary btn-small" onclick="editConfig('${id}')">Edit</button>
                        <button class="btn btn-danger btn-small" onclick="deleteConfig('${id}')">Delete</button>
                    </div>
                </div>
            `).join('');
        }
        
        // Load config selector for launcher
        async function loadConfigSelector() {
            try {
                const response = await fetch('/mcp-configs');
                const data = await response.json();
                configs = data.configs;
                renderConfigSelector();
            } catch (error) {
                console.error('Error loading configs:', error);
            }
        }
        
        // Render config selector checkboxes
        function renderConfigSelector() {
            const selector = document.getElementById('config-selector');
            if (Object.keys(configs).length === 0) {
                selector.innerHTML = '<div class="empty-state">No configurations available. Create some in the MCP Configs tab!</div>';
                return;
            }
            
            selector.innerHTML = Object.entries(configs).map(([id, config]) => `
                <label class="config-checkbox">
                    <input type="checkbox" value="${id}">
                    <div class="config-checkbox-label">
                        <div class="config-checkbox-name">${config.name}</div>
                        <div class="config-checkbox-desc">${config.description}</div>
                    </div>
                </label>
            `).join('');
        }
        
        // Modal functions
        function showCreateConfigModal() {
            document.getElementById('modal-title').textContent = 'Create MCP Configuration';
            document.getElementById('config-form').reset();
            document.getElementById('config-id').disabled = false;
            editingConfigId = null;
            document.getElementById('config-modal').style.display = 'block';
        }
        
        function editConfig(configId) {
            const config = configs[configId];
            document.getElementById('modal-title').textContent = 'Edit MCP Configuration';
            document.getElementById('config-id').value = configId;
            document.getElementById('config-id').disabled = true;
            document.getElementById('config-name').value = config.name;
            document.getElementById('config-description').value = config.description;
            document.getElementById('config-json').value = JSON.stringify(config.config, null, 2);
            editingConfigId = configId;
            document.getElementById('config-modal').style.display = 'block';
        }
        
        function closeConfigModal() {
            document.getElementById('config-modal').style.display = 'none';
        }
        
        // Config form submission
        document.getElementById('config-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const configId = document.getElementById('config-id').value;
            const configName = document.getElementById('config-name').value;
            const configDescription = document.getElementById('config-description').value;
            const configJson = document.getElementById('config-json').value;
            
            try {
                const serverConfig = JSON.parse(configJson);
                
                const payload = {
                    id: configId,
                    name: configName,
                    description: configDescription,
                    config: serverConfig
                };
                
                const url = editingConfigId ? `/mcp-configs/${editingConfigId}` : '/mcp-configs';
                const method = editingConfigId ? 'PUT' : 'POST';
                
                const response = await fetch(url, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (response.ok) {
                    closeConfigModal();
                    loadConfigs();
                    loadConfigSelector(); // Update launcher too
                    showSuccessNotification('Configuration Saved', 
                        editingConfigId ? 'Configuration updated successfully!' : 'Configuration created successfully!');
                } else {
                    const error = await response.json();
                    showErrorNotification('Configuration Error', 'Failed to save configuration', 
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.detail}</div>`);
                }
            } catch (error) {
                showErrorNotification('Invalid Configuration', 'JSON configuration is invalid', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
            }
        });
        
        // Delete configuration
        async function deleteConfig(configId) {
            if (confirm('Are you sure you want to delete this configuration?')) {
                try {
                    const response = await fetch(`/mcp-configs/${configId}`, { method: 'DELETE' });
                    if (response.ok) {
                        loadConfigs();
                        loadConfigSelector(); // Update launcher too
                        showSuccessNotification('Configuration Deleted', 'Configuration removed successfully!');
                    } else {
                        showErrorNotification('Delete Error', 'Failed to delete configuration');
                    }
                } catch (error) {
                    showErrorNotification('Delete Error', 'An error occurred while deleting', 
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
                }
            }
        }
        
        // Reload configurations from file
        async function reloadConfigs() {
            try {
                const response = await fetch('/mcp-configs/reload', { method: 'POST' });
                if (response.ok) {
                    const result = await response.json();
                    loadConfigs();
                    loadConfigSelector(); // Update launcher too
                    showSuccessNotification('Configurations Reloaded', 
                        `Successfully loaded ${result.new_count} configurations`,
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">From: ${result.config_file}</div>`);
                } else {
                    showErrorNotification('Reload Error', 'Failed to reload configurations');
                }
            } catch (error) {
                showErrorNotification('Reload Error', 'An error occurred while reloading', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
            }
        }

        // Purpose Endpoints Management
        let purposeEndpoints = {};
        let editingEndpointSlug = null;

        async function loadPurposeEndpoints() {
            try {
                const response = await fetch('/purpose/endpoints');
                const data = await response.json();
                purposeEndpoints = data.endpoints.reduce((acc, endpoint) => {
                    acc[endpoint.slug] = endpoint;
                    return acc;
                }, {});
                renderPurposeEndpoints();
            } catch (error) {
                console.error('Error loading purpose endpoints:', error);
                showErrorNotification('Load Error', 'Failed to load purpose endpoints');
            }
        }

        function renderPurposeEndpoints() {
            const grid = document.getElementById('purpose-endpoints-grid');
            if (Object.keys(purposeEndpoints).length === 0) {
                grid.innerHTML = '<div class="empty-state">No purpose endpoints found. Create your first one!</div>';
                return;
            }

            grid.innerHTML = Object.values(purposeEndpoints).map(endpoint => `
                <div class="purpose-endpoint-card">
                    <div class="endpoint-header">
                        <div>
                            <div class="endpoint-name">${endpoint.name}</div>
                            <div class="endpoint-slug-container">
                                <div class="endpoint-slug">/purpose/${endpoint.slug}</div>
                                <button class="btn-copy-url" onclick="copyEndpointUrl('${endpoint.slug}')" title="Copy full URL">
                                    📋
                                </button>
                            </div>
                        </div>
                        <div class="endpoint-status ${endpoint.enabled ? 'enabled' : 'disabled'}">
                            ${endpoint.enabled ? '✅ Enabled' : '❌ Disabled'}
                        </div>
                    </div>
                    <div class="endpoint-description">${endpoint.description}</div>
                    <div class="endpoint-meta">
                        <div class="endpoint-servers">
                            ${endpoint.mcp_servers.map(server => `<span class="server-tag">🖥️ ${server}</span>`).join('')}
                        </div>
                        <div class="endpoint-tags">
                            ${endpoint.tags.map(tag => `<span class="endpoint-tag">${tag}</span>`).join('')}
                        </div>
                    </div>
                    <div class="endpoint-actions">
                        <button class="btn btn-test btn-small" onclick="testPurposeEndpoint('${endpoint.slug}')">
                            <span>🚀</span><span>Test</span>
                        </button>
                        <button class="btn btn-edit btn-small" onclick="editPurposeEndpoint('${endpoint.slug}')">
                            <span>✏️</span><span>Edit</span>
                        </button>
                        <button class="btn btn-delete btn-small" onclick="deletePurposeEndpoint('${endpoint.slug}')">
                            <span>🗑️</span><span>Delete</span>
                        </button>
                    </div>
                </div>
            `).join('');
        }

        async function testPurposeEndpoint(slug) {
            const endpoint = purposeEndpoints[slug];
            let testData = '';
            
            // Provide example data based on endpoint
            if (slug === 'handle-github-webhook') {
                testData = JSON.stringify({
                    "webhook_data": {
                        "event_type": "push",
                        "repository": "example/repo",
                        "commits": [{"id": "abc123", "message": "Test commit"}]
                    }
                }, null, 2);
            } else if (slug === 'research-topic') {
                testData = "Latest developments in AI agent orchestration";
            } else if (slug === 'analyze-code-changes') {
                testData = JSON.stringify({
                    "file_paths": ["src/api.py", "src/models.py"],
                    "commit_message": "Add purpose endpoints feature"
                }, null, 2);
            }
            
            const inputData = prompt(`Enter test data for ${endpoint.name} (JSON or plain text):`, testData);
            if (!inputData) return;
            
            try {
                showInfoNotification('Executing Endpoint', `Testing ${endpoint.name}...`);
                
                // Try to detect if input is JSON or plain text
                let isJSON = false;
                let requestBody = inputData;
                let contentType = 'text/plain';
                
                try {
                    // Try to parse as JSON
                    const parsed = JSON.parse(inputData);
                    // If parsing succeeds, it's valid JSON
                    isJSON = true;
                    requestBody = JSON.stringify(parsed);
                    contentType = 'application/json';
                } catch (jsonError) {
                    // If parsing fails, treat as plain text
                    isJSON = false;
                    requestBody = inputData;
                    contentType = 'text/plain';
                }
                
                const response = await fetch(`/purpose/${slug}`, {
                    method: 'POST',
                    headers: { 'Content-Type': contentType },
                    body: requestBody
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showSuccessNotification('Endpoint Test Complete', 
                        `${endpoint.name} executed successfully!`,
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em; max-height: 200px; overflow-y: auto;">
                            <strong>Input Type:</strong> ${isJSON ? 'JSON' : 'Plain Text'}<br>
                            <strong>Status:</strong> ${result.status}<br>
                            <strong>Execution Time:</strong> ${result.execution_time.toFixed(2)}s<br>
                            <strong>Iterations:</strong> ${result.iterations_used}<br>
                            <strong>Result:</strong><br>
                            ${result.result.substring(0, 500)}${result.result.length > 500 ? '...' : ''}
                        </div>`);
                } else {
                    showErrorNotification('Endpoint Test Failed', 
                        `Failed to execute ${endpoint.name}`,
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${result.detail || 'Unknown error'}</div>`);
                }
            } catch (error) {
                showErrorNotification('Test Error', 'Execution error', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
            }
        }

        function editPurposeEndpoint(slug) {
            const endpoint = purposeEndpoints[slug];
            document.getElementById('purpose-modal-title').textContent = 'Edit Purpose Endpoint';
            document.getElementById('purpose-slug').value = endpoint.slug;
            document.getElementById('purpose-slug').disabled = true;
            document.getElementById('purpose-name').value = endpoint.name;
            document.getElementById('purpose-description').value = endpoint.description;
            document.getElementById('purpose-prompt').value = endpoint.prompt_template;
            document.getElementById('purpose-tags').value = endpoint.tags.join(', ');
            document.getElementById('purpose-enabled').checked = endpoint.enabled;
            
            editingEndpointSlug = slug;
            loadMcpSelectorForPurpose(endpoint.mcp_servers);
            document.getElementById('purpose-endpoint-modal').style.display = 'block';
        }

        async function deletePurposeEndpoint(slug) {
            const endpoint = purposeEndpoints[slug];
            if (confirm(`Are you sure you want to delete the purpose endpoint "${endpoint.name}"?`)) {
                try {
                    const response = await fetch(`/purpose/endpoints/${slug}`, { method: 'DELETE' });
                    if (response.ok) {
                        loadPurposeEndpoints();
                        showSuccessNotification('Endpoint Deleted', `${endpoint.name} removed successfully!`);
                    } else {
                        const error = await response.json();
                        showErrorNotification('Delete Error', 'Failed to delete endpoint',
                            `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.detail}</div>`);
                    }
                } catch (error) {
                    showErrorNotification('Delete Error', 'An error occurred while deleting', 
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
                }
            }
        }

        function showCreateEndpointModal() {
            document.getElementById('purpose-modal-title').textContent = 'Create Purpose Endpoint';
            document.getElementById('purpose-endpoint-form').reset();
            document.getElementById('purpose-slug').disabled = false;
            document.getElementById('purpose-enabled').checked = true;
            editingEndpointSlug = null;
            loadMcpSelectorForPurpose([]);
            document.getElementById('purpose-endpoint-modal').style.display = 'block';
        }

        function refreshPurposeEndpoints() {
            loadPurposeEndpoints();
        }

        function copyEndpointUrl(slug) {
            const baseUrl = window.location.origin;
            const fullUrl = `${baseUrl}/purpose/${slug}`;
            
            navigator.clipboard.writeText(fullUrl).then(() => {
                showSuccessNotification('URL Copied!', 
                    'Endpoint URL copied to clipboard',
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em; font-family: monospace;">${fullUrl}</div>`);
            }).catch(err => {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = fullUrl;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                showSuccessNotification('URL Copied!', 
                    'Endpoint URL copied to clipboard',
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em; font-family: monospace;">${fullUrl}</div>`);
            });
        }

        function closePurposeEndpointModal() {
            document.getElementById('purpose-endpoint-modal').style.display = 'none';
        }

        async function loadMcpSelectorForPurpose(selectedServers = []) {
            try {
                const response = await fetch('/mcp-configs');
                const data = await response.json();
                const mcpConfigs = data.configs;
                renderMcpSelectorForPurpose(mcpConfigs, selectedServers);
            } catch (error) {
                console.error('Error loading MCP configs for purpose selector:', error);
            }
        }

        function renderMcpSelectorForPurpose(mcpConfigs, selectedServers = []) {
            const selector = document.getElementById('purpose-mcp-selector');
            if (Object.keys(mcpConfigs).length === 0) {
                selector.innerHTML = '<div class="empty-state">No MCP configurations available. Create some in the MCP Configs tab!</div>';
                return;
            }

            selector.innerHTML = Object.entries(mcpConfigs).map(([id, config]) => `
                <label class="config-checkbox">
                    <input type="checkbox" value="${id}" ${selectedServers.includes(id) ? 'checked' : ''}>
                    <div class="config-checkbox-label">
                        <div class="config-checkbox-name">${config.name}</div>
                        <div class="config-checkbox-desc">${config.description}</div>
                    </div>
                </label>
            `).join('');
        }

        // Purpose Endpoint form submission
        document.getElementById('purpose-endpoint-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const slug = document.getElementById('purpose-slug').value;
            const name = document.getElementById('purpose-name').value;
            const description = document.getElementById('purpose-description').value;
            const promptTemplate = document.getElementById('purpose-prompt').value;
            const tagsInput = document.getElementById('purpose-tags').value;
            const enabled = document.getElementById('purpose-enabled').checked;
            
            // Parse tags
            const tags = tagsInput.split(',').map(tag => tag.trim()).filter(tag => tag.length > 0);
            
            // Get selected MCP servers
            const selectedServers = Array.from(document.querySelectorAll('#purpose-mcp-selector input[type="checkbox"]:checked'))
                .map(checkbox => checkbox.value);
            
            const payload = {
                slug: slug,
                name: name,
                description: description,
                prompt_template: promptTemplate,
                mcp_servers: selectedServers,
                tags: tags,
                enabled: enabled
            };
            
            try {
                const url = editingEndpointSlug ? `/purpose/endpoints/${editingEndpointSlug}` : '/purpose/endpoints';
                const method = editingEndpointSlug ? 'PUT' : 'POST';
                
                const response = await fetch(url, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (response.ok) {
                    closePurposeEndpointModal();
                    loadPurposeEndpoints();
                    showSuccessNotification('Purpose Endpoint Saved', 
                        editingEndpointSlug ? 'Endpoint updated successfully!' : 'Endpoint created successfully!');
                } else {
                    const error = await response.json();
                    showErrorNotification('Save Error', 'Failed to save purpose endpoint', 
                        `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.detail}</div>`);
                }
            } catch (error) {
                showErrorNotification('Save Error', 'An error occurred while saving', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
            }
        });
        
        // Launch agent
        async function launchAgent() {
            const goal = document.getElementById('launch-goal').value.trim();
            const maxIterations = parseInt(document.getElementById('launch-iterations').value);
            const timeout = parseInt(document.getElementById('launch-timeout').value);
            
            const selectedConfigs = Array.from(document.querySelectorAll('#config-selector input:checked'))
                .map(checkbox => checkbox.value);
            
            if (!goal) {
                showErrorNotification('Missing Goal', 'Please enter a goal for the agent to accomplish');
                return;
            }
            
            if (selectedConfigs.length === 0) {
                showErrorNotification('No Capabilities Selected', 'Please select at least one MCP configuration to provide capabilities to the agent');
                return;
            }
            
            try {
                const response = await fetch('/agents/launch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        goal: goal,
                        configs: selectedConfigs,
                        max_iterations: maxIterations,
                        timeout: timeout
                    })
                });
                
                const result = await response.json();
                
                // Use toast notification instead of alert
                showAgentCompletionToast(result);
                
                // Clear form
                document.getElementById('launch-goal').value = '';
                document.querySelectorAll('#config-selector input:checked').forEach(cb => cb.checked = false);
                
            } catch (error) {
                showErrorNotification('Launch Error', 'Failed to launch agent', 
                    `<div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.85em;">${error.message}</div>`);
            }
        }
        
        // Connect to orchestration stream
        const eventSource = new EventSource('/dashboard/stream');
        
        // Track last activity message to prevent duplicates
        const lastActivityMessages = {};
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('Orchestration stream data:', data);
            
            if (data.type === 'initial_state') {
                data.agents.forEach(agent => {
                    agents[agent.agent_id] = agent;
                    lastActivityMessages[agent.agent_id] = agent.current_activity;
                });
                updateOrchestration();
                addTerminalLine('System initialized with ' + data.agents.length + ' agents', 'info');
            } else if (data.type === 'agent_update') {
                const agent = data.data;
                const agentId = data.agent_id;
                console.log('Agent update for', agentId, '- details:', agent.details);
                console.log('Execution plan in details:', agent.details?.execution_plan);
                agents[agentId] = agent;
                updateOrchestration();
                
                // Only add terminal update if the activity message has changed
                const currentMessage = agent.current_activity;
                const lastMessage = lastActivityMessages[agentId];
                
                if (currentMessage !== lastMessage) {
                    const message = `[${agentId.substring(0, 8)}] ${currentMessage}`;
                    const className = agent.status === 'failed' ? 'error' : 
                                      agent.status === 'completed' ? 'success' : 'info';
                    addTerminalLine(message, className);
                    lastActivityMessages[agentId] = currentMessage;
                }
                
            } else if (data.type === 'agent_removed') {
                delete agents[data.agent_id];
                delete lastActivityMessages[data.agent_id]; // Clean up tracking
                updateOrchestration();
                addTerminalLine(`Agent ${data.agent_id.substring(0, 8)} removed`, 'info');
            }
        };
        
        function addTerminalLine(text, className = 'text') {
            const terminal = document.getElementById('terminal-output');
            const timestamp = new Date().toLocaleTimeString();
            const line = document.createElement('div');
            line.className = 'terminal-line';
            line.innerHTML = `<span class="terminal-timestamp">[${timestamp}]</span> <span class="terminal-${className}">${text}</span>`;
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
            
            // Keep only last 100 lines
            while (terminal.children.length > 100) {
                terminal.removeChild(terminal.firstChild);
            }
        }
        
        function clearTerminal() {
            document.getElementById('terminal-output').innerHTML = '';
            addTerminalLine('Terminal cleared', 'info');
        }
        
        function updateOrchestration() {
            const agentList = Object.values(agents);
            const activeAgents = agentList.filter(a => a.status === 'running' || a.status === 'pending');
            const completedAgents = agentList.filter(a => a.status === 'completed');
            
            updateAgentContainer('active-agents', activeAgents);
            updateAgentContainer('completed-agents', completedAgents);
        }
        
        function updateAgentContainer(containerId, agentList) {
            const container = document.getElementById(containerId);
            
            if (agentList.length === 0) {
                container.innerHTML = `<div class="empty-state">No ${containerId.includes('active') ? 'active' : 'completed'} agents</div>`;
                return;
            }
            
            container.innerHTML = agentList.map(agent => `
                <div class="agent-card ${agent.status}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                        <div style="font-family: monospace; font-size: 0.85em; background: #f1f5f9; color: #4a5568; padding: 4px 8px; border-radius: 4px;">
                            ${agent.agent_id.substring(0, 8)}
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <div style="background: ${agent.status === 'running' ? '#dcfce7' : agent.status === 'completed' ? '#dbeafe' : '#fee2e2'}; 
                                        color: ${agent.status === 'running' ? '#166534' : agent.status === 'completed' ? '#1d4ed8' : '#dc2626'}; 
                                        padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">
                                ${agent.status}
                            </div>
                            ${agent.status === 'completed' ? 
                                `<button class="btn btn-secondary btn-small" onclick="showAgentDetails('${agent.agent_id}')">Details</button>` 
                                : ''}
                        </div>
                    </div>
                    <div style="font-weight: 600; margin-bottom: 10px; line-height: 1.3; color: #1a202c;">${agent.goal}</div>
                    
                    ${agent.status === 'running' || agent.status === 'pending' ? `
                        <div class="current-activity">
                            <strong style="color: #2d3748;">🔄 Current:</strong> <span style="color: #4a5568;">${agent.current_activity}</span>
                        </div>
                        ${agent.execution_plan && agent.execution_plan.length > 0 ? `
                            <div class="execution-plan">
                                <div class="execution-plan-title">
                                    <span style="color: #2d3748;">📋 Execution Plan</span>
                                    ${agent.plan_updated ? '<span class="plan-updated-badge">UPDATED</span>' : ''}
                                </div>
                                <ul class="execution-plan-steps">
                                    ${agent.execution_plan.map(step => `
                                        <li class="execution-plan-step">
                                            <span class="step-indicator">${step.substring(0, 2)}</span>
                                            <span class="step-text" style="color: #4a5568;">${step.substring(2).trim()}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        ${agent.recent_activities && agent.recent_activities.length > 0 ? `
                            <div class="recent-activities">
                                <strong style="font-size: 0.85em; color: #4a5568;">Recent Activity:</strong>
                                ${agent.recent_activities.slice(-3).reverse().map(activity => `
                                    <div class="activity-item" style="color: #718096;">
                                        <span class="activity-timestamp">${new Date(activity.timestamp).toLocaleTimeString()}</span>
                                        - ${activity.message}
                                    </div>
                                `).join('')}
                            </div>
                        ` : ''}
                    ` : ''}
                    
                    ${agent.result ? `
                        <div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px; margin: 10px 0; border-radius: 4px;">
                            <div style="font-weight: 600; color: #1e40af; font-size: 0.9em; margin-bottom: 5px;">RESULT</div>
                            <div style="color: #374151; font-size: 0.95em; max-height: 100px; overflow-y: auto;">${agent.result}</div>
                        </div>
                    ` : ''}
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 10px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: #2d3748;">${agent.iterations_used}</div>
                            <div style="font-size: 0.75em; color: #718096;">Iterations</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: #2d3748;">${agent.execution_time > 0 ? agent.execution_time.toFixed(1) + 's' : 'Running...'}</div>
                            <div style="font-size: 0.75em; color: #718096;">Time</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: #2d3748;">${agent.progress.toFixed(0)}%</div>
                            <div style="font-size: 0.75em; color: #718096;">Progress</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        // Agent details modal functions
        function showAgentDetails(agentId) {
            const agent = agents[agentId];
            if (!agent) return;
            
            document.getElementById('details-title').textContent = `Agent ${agentId.substring(0, 8)} Details`;
            
            // Overview tab
            document.getElementById('details-overview-content').innerHTML = `
                <div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                        <div>
                            <h4 style="color: #2d3748;">Basic Information</h4>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Agent ID:</strong> <code style="color: #2d3748;">${agent.agent_id}</code></p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Status:</strong> <span style="color: ${agent.status === 'completed' ? '#1d4ed8' : '#dc2626'}">${agent.status.toUpperCase()}</span></p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Goal:</strong> ${agent.goal}</p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Start Time:</strong> ${new Date(agent.start_time * 1000).toLocaleString()}</p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Last Update:</strong> ${new Date(agent.last_update * 1000).toLocaleString()}</p>
                        </div>
                        <div>
                            <h4 style="color: #2d3748;">Performance Metrics</h4>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Execution Time:</strong> ${agent.execution_time.toFixed(2)} seconds</p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Iterations Used:</strong> ${agent.iterations_used}</p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Progress:</strong> ${agent.progress.toFixed(1)}%</p>
                            <p style="color: #4a5568;"><strong style="color: #2d3748;">Current Activity:</strong> ${agent.current_activity}</p>
                            ${agent.selected_configs ? `<p style="color: #4a5568;"><strong style="color: #2d3748;">MCP Configs:</strong> ${agent.selected_configs.join(', ')}</p>` : ''}
                        </div>
                    </div>
                    ${agent.details && agent.details.execution_plan && agent.details.execution_plan.length > 0 ? `
                        <div style="margin-top: 20px;">
                            <h4 style="color: #2d3748;">📋 Execution Plan ${agent.plan_updated ? '<span class="plan-updated-badge">UPDATED</span>' : ''}</h4>
                            <div class="execution-plan" style="margin-top: 10px;">
                                <ul class="execution-plan-steps">
                                    ${agent.execution_plan.map(step => `
                                        <li class="execution-plan-step">
                                            <span class="step-indicator">${step.substring(0, 2)}</span>
                                            <span class="step-text" style="color: #4a5568;">${step.substring(2).trim()}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
            
            // Execution log tab
            const executionLog = agent.recent_activities ? 
                agent.recent_activities.map(activity => 
                    `[${new Date(activity.timestamp).toLocaleTimeString()}] ${activity.message}${activity.progress ? ` (${activity.progress.toFixed(1)}%)` : ''}`
                ).join('\\n') : 
                'No execution log available';
            document.getElementById('details-execution-log').textContent = executionLog;
            
            // Result tab
            document.getElementById('details-result-content').innerHTML = `
                <div style="background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <h4 style="color: #1e40af; margin-bottom: 10px;">Agent Result</h4>
                    <div style="white-space: pre-wrap; line-height: 1.6; font-size: 0.95em; color: #2d3748;">${agent.result || 'No result available'}</div>
                </div>
            `;
            
            // Config tab
            const configDetails = agent.selected_configs ? 
                agent.selected_configs.map(configId => {
                    const config = configs[configId];
                    return config ? `
                        <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h5 style="color: #2d3748;">${config.name}</h5>
                            <p style="color: #666; margin: 5px 0;">${config.description}</p>
                            <pre style="background: #e2e8f0; padding: 10px; border-radius: 4px; font-size: 0.8em; overflow-x: auto; color: #2d3748;">${JSON.stringify(config.config, null, 2)}</pre>
                        </div>
                    ` : `<p style="color: #2d3748;">Config ${configId} not found</p>`;
                }).join('') :
                '<div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;"><p style="color: #2d3748;">No configuration details available</p></div>';
            document.getElementById('details-config-content').innerHTML = `
                <div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    ${configDetails}
                </div>
            `;
            
            // Plan tab
            const planContent = agent.execution_plan && agent.execution_plan.length > 0 ? `
                <div style="margin-bottom: 15px;">
                    <p style="color: #666; font-size: 0.9em;">
                        Real-time execution plan with progress indicators. 
                        ${agent.plan_updated ? '<strong style="color: #3b82f6;">Plan was updated during execution.</strong>' : ''}
                    </p>
                </div>
                <div class="execution-plan" style="background: white; border: 2px solid #e2e8f0; color: #2d3748;">
                    <ul class="execution-plan-steps" style="font-size: 1em;">
                        ${agent.execution_plan.map(step => `
                            <li class="execution-plan-step" style="padding: 8px 0; border-bottom: 1px solid #f1f5f9; color: #2d3748;">
                                <span class="step-indicator" style="font-size: 1.2em;">${step.substring(0, 2)}</span>
                                <span class="step-text" style="font-size: 0.95em; color: #4a5568;">${step.substring(2).trim()}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: #f8fafc; border-radius: 6px; font-size: 0.85em; color: #666;">
                    <strong style="color: #2d3748;">Legend:</strong> 
                    ✅ Completed • ❌ Failed • 🔄 In Progress • ⏳ Pending
                </div>
            ` : `
                <div style="text-align: center; color: #a0aec0; font-style: italic; padding: 40px 20px;">
                    <div style="font-size: 3em; margin-bottom: 10px;">📋</div>
                    <p style="color: #a0aec0;">No execution plan available for this agent.</p>
                    <p style="font-size: 0.85em; margin-top: 10px; color: #a0aec0;">The agent may not have generated a structured plan, or it was created before plan tracking was implemented.</p>
                </div>
            `;
            document.getElementById('details-plan-content').innerHTML = planContent;
            
            document.getElementById('details-modal').style.display = 'block';
        }
        
        function closeDetailsModal() {
            document.getElementById('details-modal').style.display = 'none';
        }
        
        function showDetailsTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.details-tab').forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.details-content').forEach(content => content.classList.remove('active'));
            document.getElementById(`details-${tabName}`).classList.add('active');
        }
        
        // Initialize
        loadConfigSelector();
        loadConfigs();
        
        // Set initial timestamp to current time
        document.getElementById('initial-timestamp').textContent = '[' + new Date().toLocaleTimeString() + ']';
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# MCP Configuration Management Endpoints

@app.get("/mcp-configs")
async def get_mcp_configs():
    """Get all MCP configurations"""
    return {
        "configs": mcp_configs,
        "count": len(mcp_configs)
    }

@app.get("/mcp-configs/info")
async def get_mcp_config_info():
    """Get information about the MCP configuration file"""
    return {
        "config_file": str(CONFIG_FILE),
        "config_dir": str(CONFIG_DIR),
        "file_exists": CONFIG_FILE.exists(),
        "config_count": len(mcp_configs),
        "last_modified": CONFIG_FILE.stat().st_mtime if CONFIG_FILE.exists() else None
    }

@app.get("/mcp-configs/{config_id}")
async def get_mcp_config(config_id: str):
    """Get a specific MCP configuration"""
    if config_id not in mcp_configs:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return mcp_configs[config_id]

@app.post("/mcp-configs")
async def create_mcp_config(request: Request):
    """Create a new MCP configuration"""
    try:
        data = await request.json()
        
        config_id = data.get("id")
        if not config_id:
            raise HTTPException(status_code=400, detail="Configuration ID is required")
        
        if config_id in mcp_configs:
            raise HTTPException(status_code=409, detail="Configuration ID already exists")
        
        # Validate required fields
        required_fields = ["name", "description", "config"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Field '{field}' is required")
        
        # Validate the MCP server config structure
        server_config = data["config"]
        server_required = ["name", "command", "args", "transport"]
        for field in server_required:
            if field not in server_config:
                raise HTTPException(status_code=400, detail=f"Server config field '{field}' is required")
        
        mcp_configs[config_id] = {
            "name": data["name"],
            "description": data["description"], 
            "config": server_config
        }
        
        # Save to file
        if not save_mcp_configs():
            logger.warning("Failed to save MCP configurations to file")
        
        return {
            "message": "Configuration created successfully",
            "config_id": config_id,
            "config": mcp_configs[config_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating MCP config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/mcp-configs/{config_id}")
async def update_mcp_config(config_id: str, request: Request):
    """Update an existing MCP configuration"""
    try:
        if config_id not in mcp_configs:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        data = await request.json()
        
        # Update the configuration
        if "name" in data:
            mcp_configs[config_id]["name"] = data["name"]
        if "description" in data:
            mcp_configs[config_id]["description"] = data["description"]
        if "config" in data:
            # Validate the server config structure
            server_config = data["config"]
            server_required = ["name", "command", "args", "transport"]
            for field in server_required:
                if field not in server_config:
                    raise HTTPException(status_code=400, detail=f"Server config field '{field}' is required")
            mcp_configs[config_id]["config"] = server_config
        
        # Save to file
        if not save_mcp_configs():
            logger.warning("Failed to save MCP configurations to file")
        
        return {
            "message": "Configuration updated successfully",
            "config_id": config_id,
            "config": mcp_configs[config_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MCP config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/mcp-configs/{config_id}")
async def delete_mcp_config(config_id: str):
    """Delete an MCP configuration"""
    if config_id not in mcp_configs:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    deleted_config = mcp_configs.pop(config_id)
    
    # Save to file
    if not save_mcp_configs():
        logger.warning("Failed to save MCP configurations to file")
    
    return {
        "message": "Configuration deleted successfully",
        "config_id": config_id,
        "deleted_config": deleted_config
    }

@app.post("/mcp-configs/reload")
async def reload_mcp_configs():
    """Reload MCP configurations from the JSON file"""
    try:
        old_count = len(mcp_configs)
        load_mcp_configs()
        new_count = len(mcp_configs)
        
        return {
            "message": "Configurations reloaded successfully",
            "old_count": old_count,
            "new_count": new_count,
            "config_file": str(CONFIG_FILE),
            "configs": mcp_configs
        }
    except Exception as e:
        logger.error(f"Error reloading MCP configurations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload configurations: {str(e)}")

@app.post("/agents/launch")
async def launch_agent_with_configs(request: Request):
    """Launch an agent with selected MCP configurations"""
    try:
        data = await request.json()
        goal = data.get("goal", "").strip()
        selected_configs = data.get("configs", [])
        max_iterations = data.get("max_iterations", 15)
        timeout = data.get("timeout", 180)
        
        if not goal:
            return {
                "result": "Error: No goal provided",
                "details": "Please provide a goal for the agent",
                "execution_time": 0,
                "iterations_used": 0,
                "status": "failed",
                "agent_id": None
            }
        
        if not selected_configs:
            return {
                "result": "Error: No configurations selected",
                "details": "Please select at least one MCP configuration",
                "execution_time": 0,
                "iterations_used": 0,
                "status": "failed",
                "agent_id": None
            }
        
        # Build server list from selected configurations
        servers = []
        for config_id in selected_configs:
            if config_id not in mcp_configs:
                return {
                    "result": f"Error: Configuration '{config_id}' not found",
                    "details": f"Selected configuration '{config_id}' does not exist",
                    "execution_time": 0,
                    "iterations_used": 0,
                    "status": "failed",
                    "agent_id": None
                }
            
            server_config = mcp_configs[config_id]["config"]
            servers.append(MCPServerConfig(**server_config))
        
        # Create and run the agent
        agent_id = str(uuid.uuid4())
        
        # Initialize dashboard data
        orchestration_data[agent_id] = {
            "agent_id": agent_id,
            "goal": goal,
            "status": "pending",
            "current_activity": "Initializing...",
            "progress": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "iterations_used": 0,
            "details": {},
            "recent_activities": [],
            "execution_time": 0,
            "result": "",
            "selected_configs": selected_configs
        }
        
        config = MCPAgentConfig(
            servers=servers,
            goal=AgentGoal(
                description=goal,
                success_criteria="Provide comprehensive and helpful response",
                context={"launch_method": "ui", "selected_configs": selected_configs}
            ),
            max_iterations=max_iterations,
            timeout=timeout
        )
        
        agent_request = AgentRequest(id=agent_id, config=config)
        
        # Collect execution details
        execution_details = []
        
        async def detail_callback(update: StatusUpdate):
            """Collect detailed execution steps and update orchestration"""
            # Update orchestration data (this is the key fix!)
            await status_callback(update)
            timestamp = update.timestamp.strftime("%H:%M:%S")
            progress = f" ({update.progress:.1f}%)" if update.progress else ""
            detail_line = f"[{timestamp}] {update.message}{progress}"
            if update.details:
                if isinstance(update.details, dict):
                    details_str = ", ".join([f"{k}: {v}" for k, v in update.details.items()])
                    detail_line += f" - {details_str}"
                else:
                    detail_line += f" - {update.details}"
            execution_details.append(detail_line)
        
        config_names = [mcp_configs[c]["name"] for c in selected_configs]
        logger.info(f"Launching agent with configs: {config_names} for goal: '{goal[:50]}...'")
        
        # Run the agent
        response = await orchestrator.run_single_agent(agent_request, detail_callback)
        
        logger.info(f"Orchestrator response status: {response.status} (type: {type(response.status)})")
        
        # Update orchestration with final result BEFORE returning response
        if agent_id in orchestration_data:
            orchestration_data[agent_id]["result"] = response.result
            orchestration_data[agent_id]["iterations_used"] = response.iterations_used
            orchestration_data[agent_id]["execution_time"] = response.execution_time
            orchestration_data[agent_id]["status"] = response.status.value
            orchestration_data[agent_id]["progress"] = 100.0
        
        details_text = "\n".join(execution_details)
        
        return {
            "result": response.result,
            "details": details_text,
            "execution_time": response.execution_time,
            "iterations_used": response.iterations_used,
            "status": response.status.value,
            "agent_id": response.agent_id,
            "configs_used": config_names
        }
        
    except Exception as e:
        logger.error(f"Agent launch error: {e}")
        return {
            "result": "Error occurred during agent launch",
            "details": f"Error: {str(e)}",
            "execution_time": 0,
            "iterations_used": 0,
            "status": "failed",
            "agent_id": None,
            "configs_used": []
        }

# Purpose Endpoint Management
@app.get("/purpose/endpoints")
async def list_purpose_endpoints():
    """List all available purpose endpoints"""
    endpoints = purpose_manager.list_endpoints()
    return {
        "endpoints": [endpoint.dict() for endpoint in endpoints],
        "count": len(endpoints)
    }

@app.get("/purpose/endpoints/{slug}")
async def get_purpose_endpoint(slug: str):
    """Get a specific purpose endpoint"""
    endpoint = purpose_manager.get_endpoint(slug)
    if not endpoint:
        raise HTTPException(status_code=404, detail=f"Purpose endpoint '{slug}' not found")
    return endpoint.dict()

@app.post("/purpose/endpoints")
async def create_purpose_endpoint(request: Request):
    """Create a new purpose endpoint"""
    try:
        data = await request.json()
        config = PurposeEndpointConfig(**data)
        
        if purpose_manager.create_endpoint(config):
            return {"message": f"Purpose endpoint '{config.slug}' created successfully", "endpoint": config.dict()}
        else:
            raise HTTPException(status_code=500, detail="Failed to create purpose endpoint")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/purpose/endpoints/{slug}")
async def update_purpose_endpoint(slug: str, request: Request):
    """Update an existing purpose endpoint"""
    try:
        data = await request.json()
        config = PurposeEndpointConfig(**data)
        
        if purpose_manager.update_endpoint(slug, config):
            return {"message": f"Purpose endpoint '{slug}' updated successfully", "endpoint": config.dict()}
        else:
            raise HTTPException(status_code=500, detail="Failed to update purpose endpoint")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/purpose/endpoints/{slug}")
async def delete_purpose_endpoint(slug: str):
    """Delete a purpose endpoint"""
    if purpose_manager.delete_endpoint(slug):
        return {"message": f"Purpose endpoint '{slug}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Purpose endpoint '{slug}' not found")

# Purpose Endpoint Execution
@app.post("/purpose/{slug}")
async def execute_purpose_endpoint(slug: str, request: Request):
    """Execute a purpose endpoint by slug"""
    try:
        # Get input data - handle both JSON and plain text
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            # Parse as JSON
            input_data = await request.json()
        else:
            # Treat as plain text
            body_bytes = await request.body()
            input_data = body_bytes.decode('utf-8') if body_bytes else ""
        
        # Create purpose endpoint request
        purpose_request = PurposeEndpointRequest(
            endpoint_slug=slug,
            input_data=input_data
        )
        
        # Get endpoint config for dashboard display
        endpoint_config = purpose_manager.get_endpoint(slug)
        if not endpoint_config:
            raise HTTPException(status_code=404, detail=f"Purpose endpoint '{slug}' not found")
        
        # Generate agent ID for dashboard tracking
        agent_id = purpose_request.request_id
        
        # Initialize orchestration data for dashboard (matching existing pattern)
        orchestration_data[agent_id] = {
            "agent_id": agent_id,
            "goal": f"Purpose: {endpoint_config.name} - {str(input_data)[:50]}{'...' if len(str(input_data)) > 50 else ''}",
            "status": "running",
            "current_activity": f"Starting purpose endpoint: {endpoint_config.name}",
            "progress": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "iterations_used": 0,
            "details": {
                "endpoint_slug": slug,
                "endpoint_name": endpoint_config.name,
                "endpoint_description": endpoint_config.description,
                "mcp_servers": endpoint_config.mcp_servers,
                "input_preview": str(input_data)[:100] + "..." if len(str(input_data)) > 100 else str(input_data),
                "type": "purpose_endpoint"
            },
            "recent_activities": [],
            "execution_time": 0,
            "result": ""
        }
        
        # Broadcast initial state to dashboard
        await broadcast_orchestration_update({
            "type": "agent_update",
            "agent_id": agent_id,
            "data": orchestration_data[agent_id]
        })
        
        # Create status callback for dashboard integration
        async def dashboard_callback(update: StatusUpdate):
            """Update dashboard with purpose endpoint progress"""
            await status_callback(update)
        
        # Execute the endpoint with dashboard integration
        response = await purpose_manager.execute_endpoint(purpose_request, dashboard_callback)
        
        # Update orchestration data with final result
        if agent_id in orchestration_data:
            orchestration_data[agent_id]["result"] = response.result
            orchestration_data[agent_id]["iterations_used"] = response.iterations_used
            orchestration_data[agent_id]["execution_time"] = response.execution_time
            orchestration_data[agent_id]["status"] = response.status.value
            orchestration_data[agent_id]["progress"] = 100
            orchestration_data[agent_id]["current_activity"] = f"Purpose endpoint completed: {response.status.value}"
            orchestration_data[agent_id]["last_update"] = time.time()
            
            # Broadcast completion to dashboard
            await broadcast_orchestration_update({
                "type": "agent_update",
                "agent_id": agent_id,
                "data": orchestration_data[agent_id]
            })
        
        return response.dict()
        
    except Exception as e:
        # Update orchestration data with error if agent_id exists
        if 'agent_id' in locals() and agent_id in orchestration_data:
            orchestration_data[agent_id]["status"] = "failed"
            orchestration_data[agent_id]["result"] = f"Error: {str(e)}"
            orchestration_data[agent_id]["progress"] = 100
            orchestration_data[agent_id]["current_activity"] = f"Purpose endpoint failed: {str(e)}"
            orchestration_data[agent_id]["last_update"] = time.time()
            orchestration_data[agent_id]["execution_time"] = time.time() - orchestration_data[agent_id]["start_time"]
            
            await broadcast_orchestration_update({
                "type": "agent_update",
                "agent_id": agent_id,
                "data": orchestration_data[agent_id]
            })
        
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/purpose/{slug}/stream")
async def execute_purpose_endpoint_stream(slug: str, request: Request):
    """Execute a purpose endpoint with streaming updates"""
    try:
        # Get input data - handle both JSON and plain text
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            # Parse as JSON
            input_data = await request.json()
        else:
            # Treat as plain text
            body_bytes = await request.body()
            input_data = body_bytes.decode('utf-8') if body_bytes else ""
        
        # Create purpose endpoint request
        purpose_request = PurposeEndpointRequest(
            endpoint_slug=slug,
            input_data=input_data
        )
        
        # Create status callback for streaming
        status_updates = []
        
        async def stream_callback(update: StatusUpdate):
            status_updates.append(update)
            # In a real implementation, you'd stream this to the client
        
        # Execute the endpoint
        response = await purpose_manager.execute_endpoint(purpose_request, stream_callback)
        
        return {
            "response": response.dict(),
            "status_updates": [update.dict() for update in status_updates]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# MCP Server Mode
@app.post("/mcp/server/start")
async def start_mcp_server_mode():
    """
    Start TELOSCRIPT in MCP server mode
    
    This endpoint initiates TELOSCRIPT as an MCP server that can be used by other agents
    for recursive orchestration.
    """
    try:
        from .mcp_server import TeloscriptMCPServer
        
        # Create and start the MCP server
        mcp_server = TeloscriptMCPServer(teloscript_api_url="http://localhost:8000")
        
        # Note: In a real implementation, this would run in a separate process
        # For now, we'll return instructions on how to run the MCP server
        return {
            "message": "MCP Server mode available",
            "instructions": [
                "To run TELOSCRIPT as an MCP server, use the standalone script:",
                "python teloscript_mcp_server.py",
                "",
                "Or configure it in your MCP client configuration:",
                {
                    "name": "teloscript",
                    "command": "python",
                    "args": ["/path/to/teloscript_mcp_server.py"],
                    "transport": "stdio"
                }
            ],
            "available_tools": [
                "spawn_agent - Create and execute a new TELOSCRIPT agent",
                "check_agent_status - Check the status of a running agent", 
                "execute_purpose_endpoint - Execute a predefined purpose endpoint",
                "list_purpose_endpoints - List all available purpose endpoints",
                "get_agent_capabilities - List available MCP servers and their capabilities",
                "create_workflow_template - Create a reusable workflow template",
                "list_active_agents - List all currently active agents",
                "cancel_agent - Cancel a running agent"
            ],
            "mcp_server_config": {
                "name": "teloscript",
                "command": "python",
                "args": ["/path/to/teloscript_mcp_server.py"],
                "transport": "stdio",
                "description": "TELOSCRIPT MCP Server for recursive agent orchestration"
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting MCP server mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start MCP server mode: {str(e)}")

@app.get("/mcp/server/info")
async def get_mcp_server_info():
    """Get information about TELOSCRIPT's MCP server capabilities"""
    return {
        "server_name": "teloscript",
        "server_version": "1.0.0",
        "description": "TELOSCRIPT as an MCP Server for recursive agent orchestration",
        "capabilities": {
            "tools": [
                {
                    "name": "spawn_agent",
                    "description": "Create and execute a new TELOSCRIPT agent",
                    "parameters": ["goal", "mcp_servers", "max_iterations", "timeout"]
                },
                {
                    "name": "check_agent_status", 
                    "description": "Check the status of a running agent",
                    "parameters": ["agent_id"]
                },
                {
                    "name": "execute_purpose_endpoint",
                    "description": "Execute a predefined purpose endpoint",
                    "parameters": ["endpoint_slug", "input_data"]
                },
                {
                    "name": "list_purpose_endpoints",
                    "description": "List all available purpose endpoints",
                    "parameters": []
                },
                {
                    "name": "get_agent_capabilities",
                    "description": "List available MCP servers and their capabilities", 
                    "parameters": []
                },
                {
                    "name": "create_workflow_template",
                    "description": "Create a reusable workflow template",
                    "parameters": ["name", "description", "steps", "input_schema"]
                },
                {
                    "name": "list_active_agents",
                    "description": "List all currently active agents",
                    "parameters": []
                },
                {
                    "name": "cancel_agent",
                    "description": "Cancel a running agent",
                    "parameters": ["agent_id"]
                }
            ]
        },
        "usage_examples": [
            {
                "scenario": "Spawn a research agent",
                "tool": "spawn_agent",
                "parameters": {
                    "goal": "Research the latest AI developments and create a summary report",
                    "mcp_servers": ["brave-search", "filesystem"],
                    "max_iterations": 15
                }
            },
            {
                "scenario": "Execute purpose endpoint",
                "tool": "execute_purpose_endpoint", 
                "parameters": {
                    "endpoint_slug": "handle-github-webhook",
                    "input_data": {"repository": "user/repo", "action": "push"}
                }
            }
        ],
        "installation": {
            "standalone_script": "python teloscript_mcp_server.py",
            "mcp_config": {
                "name": "teloscript",
                "command": "python",
                "args": ["/path/to/teloscript_mcp_server.py"],
                "transport": "stdio"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 