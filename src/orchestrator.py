import asyncio
from typing import Dict, List, Optional, Callable
from loguru import logger
import uuid
import sys
import os

# Fix imports
try:
    from .models import (
        MCPAgentConfig, AgentRequest, AgentResponse, 
        StatusUpdate, AgentStatus
    )
    from .mcp_agent import LLMPoweredMCPAgent as MCPAgent
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from models import (
        MCPAgentConfig, AgentRequest, AgentResponse, 
        StatusUpdate, AgentStatus
    )
    from mcp_agent import LLMPoweredMCPAgent as MCPAgent

class WorkflowOrchestrator:
    """Orchestrates multiple MCP agents in sequential or parallel workflows"""
    
    def __init__(self):
        self.active_agents: Dict[str, MCPAgent] = {}
        self.status_callbacks: Dict[str, Callable] = {}
    
    async def run_single_agent(self, request: AgentRequest, status_callback: Optional[Callable] = None) -> AgentResponse:
        """Run a single MCP agent"""
        agent = MCPAgent(request.id, request.config, status_callback)
        self.active_agents[request.id] = agent
        
        try:
            response = await agent.run()
            return response
        finally:
            # Clean up
            if request.id in self.active_agents:
                del self.active_agents[request.id]
            if status_callback and request.id in self.status_callbacks:
                del self.status_callbacks[request.id]
    
    async def run_sequential_workflow(self, requests: List[AgentRequest], status_callback: Optional[Callable] = None) -> List[AgentResponse]:
        """Run multiple agents sequentially"""
        workflow_id = str(uuid.uuid4())
        responses = []
        
        if status_callback:
            await status_callback(StatusUpdate(
                agent_id=workflow_id,
                status=AgentStatus.RUNNING,
                message=f"Starting sequential workflow with {len(requests)} agents"
            ))
        
        for i, request in enumerate(requests):
            try:
                if status_callback:
                    await status_callback(StatusUpdate(
                        agent_id=workflow_id,
                        status=AgentStatus.RUNNING,
                        message=f"Running agent {i+1}/{len(requests)}: {request.id}",
                        progress=(i / len(requests)) * 100
                    ))
                
                response = await self.run_single_agent(request, status_callback)
                responses.append(response)
                
                # If an agent fails and it's critical, stop the workflow
                if response.status == AgentStatus.FAILED:
                    logger.warning(f"Agent {request.id} failed, continuing workflow")
                
            except Exception as e:
                logger.error(f"Error running agent {request.id}: {e}")
                # Create error response
                error_response = AgentResponse(
                    agent_id=request.id,
                    status=AgentStatus.FAILED,
                    result="",
                    execution_time=0,
                    iterations_used=0,
                    status_updates=[],
                    error=str(e)
                )
                responses.append(error_response)
        
        if status_callback:
            successful_agents = len([r for r in responses if r.status == AgentStatus.COMPLETED])
            await status_callback(StatusUpdate(
                agent_id=workflow_id,
                status=AgentStatus.COMPLETED,
                message=f"Sequential workflow completed: {successful_agents}/{len(requests)} agents successful",
                progress=100
            ))
        
        return responses
    
    async def run_parallel_workflow(self, requests: List[AgentRequest], status_callback: Optional[Callable] = None) -> List[AgentResponse]:
        """Run multiple agents in parallel"""
        workflow_id = str(uuid.uuid4())
        
        if status_callback:
            await status_callback(StatusUpdate(
                agent_id=workflow_id,
                status=AgentStatus.RUNNING,
                message=f"Starting parallel workflow with {len(requests)} agents"
            ))
        
        # Create tasks for all agents
        tasks = []
        for request in requests:
            task = asyncio.create_task(
                self.run_single_agent(request, status_callback),
                name=f"agent-{request.id}"
            )
            tasks.append(task)
        
        # Wait for all agents to complete
        responses = []
        completed_tasks = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                responses.append(response)
                completed_tasks += 1
                
                if status_callback:
                    await status_callback(StatusUpdate(
                        agent_id=workflow_id,
                        status=AgentStatus.RUNNING,
                        message=f"Agent completed: {completed_tasks}/{len(requests)}",
                        progress=(completed_tasks / len(requests)) * 100
                    ))
                
            except Exception as e:
                logger.error(f"Error in parallel agent execution: {e}")
                completed_tasks += 1
                # Create error response
                error_response = AgentResponse(
                    agent_id="unknown",
                    status=AgentStatus.FAILED,
                    result="",
                    execution_time=0,
                    iterations_used=0,
                    status_updates=[],
                    error=str(e)
                )
                responses.append(error_response)
        
        if status_callback:
            successful_agents = len([r for r in responses if r.status == AgentStatus.COMPLETED])
            await status_callback(StatusUpdate(
                agent_id=workflow_id,
                status=AgentStatus.COMPLETED,
                message=f"Parallel workflow completed: {successful_agents}/{len(requests)} agents successful",
                progress=100
            ))
        
        return responses
    
    async def cancel_agent(self, agent_id: str) -> bool:
        """Cancel a running agent"""
        if agent_id in self.active_agents:
            await self.active_agents[agent_id].cancel()
            return True
        return False
    
    async def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get the current status of an agent"""
        if agent_id in self.active_agents:
            return self.active_agents[agent_id].status
        return None
    
    def get_active_agents(self) -> List[str]:
        """Get list of currently active agent IDs"""
        return list(self.active_agents.keys())
    
    async def cancel_all_agents(self) -> int:
        """Cancel all running agents"""
        cancelled_count = 0
        for agent_id in list(self.active_agents.keys()):
            if await self.cancel_agent(agent_id):
                cancelled_count += 1
        return cancelled_count 