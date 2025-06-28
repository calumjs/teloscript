#!/usr/bin/env python3
"""
Example MCP Client for TELOSCRIPT MCP Server

This example demonstrates how to use TELOSCRIPT as an MCP server from another agent or client.
"""

import asyncio
import json
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class TeloscriptMCPClient:
    """Example client for interacting with TELOSCRIPT MCP Server"""
    
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session = None
    
    async def connect(self):
        """Connect to the TELOSCRIPT MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path]
        )
        
        self.session = await stdio_client(server_params).__aenter__()
        
        # Initialize the session
        init_result = await self.session.initialize()
        print(f"Connected to TELOSCRIPT MCP Server: {init_result.server_name}")
        
        # List available tools
        tools = await self.session.list_tools()
        print(f"Available tools: {[tool.name for tool in tools.tools]}")
        
        return self.session
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
    
    async def spawn_agent(self, goal: str, mcp_servers: list = None, max_iterations: int = 10, timeout: int = 300) -> Dict[str, Any]:
        """Spawn a new TELOSCRIPT agent"""
        result = await self.session.call_tool(
            name="spawn_agent",
            arguments={
                "goal": goal,
                "mcp_servers": mcp_servers or ["filesystem"],
                "max_iterations": max_iterations,
                "timeout": timeout
            }
        )
        return json.loads(result.content[0].text)
    
    async def check_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Check the status of an agent"""
        result = await self.session.call_tool(
            name="check_agent_status",
            arguments={"agent_id": agent_id}
        )
        return json.loads(result.content[0].text)
    
    async def execute_purpose_endpoint(self, endpoint_slug: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a purpose endpoint"""
        result = await self.session.call_tool(
            name="execute_purpose_endpoint",
            arguments={
                "endpoint_slug": endpoint_slug,
                "input_data": input_data
            }
        )
        return json.loads(result.content[0].text)
    
    async def list_purpose_endpoints(self) -> Dict[str, Any]:
        """List all available purpose endpoints"""
        result = await self.session.call_tool(
            name="list_purpose_endpoints",
            arguments={}
        )
        return json.loads(result.content[0].text)
    
    async def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get available MCP servers and capabilities"""
        result = await self.session.call_tool(
            name="get_agent_capabilities",
            arguments={}
        )
        return json.loads(result.content[0].text)
    
    async def list_active_agents(self) -> Dict[str, Any]:
        """List all active agents"""
        result = await self.session.call_tool(
            name="list_active_agents",
            arguments={}
        )
        return json.loads(result.content[0].text)
    
    async def cancel_agent(self, agent_id: str) -> Dict[str, Any]:
        """Cancel an agent"""
        result = await self.session.call_tool(
            name="cancel_agent",
            arguments={"agent_id": agent_id}
        )
        return json.loads(result.content[0].text)


async def example_1_simple_research():
    """Example 1: Simple research task"""
    print("\n=== Example 1: Simple Research Task ===")
    
    client = TeloscriptMCPClient("/path/to/teloscript_mcp_server.py")
    
    try:
        await client.connect()
        
        # Spawn a research agent
        result = await client.spawn_agent(
            goal="Research the latest developments in AI and machine learning",
            mcp_servers=["brave-search", "filesystem"],
            max_iterations=10
        )
        
        if result.get("success"):
            agent_id = result.get("agent_id")
            print(f"✓ Research agent spawned: {agent_id}")
            
            # Check status (in a real scenario, you'd poll this)
            status = await client.check_agent_status(agent_id)
            print(f"✓ Agent status: {status}")
        else:
            print(f"✗ Failed to spawn agent: {result.get('error')}")
    
    finally:
        await client.disconnect()


async def example_2_chained_operations():
    """Example 2: Chained operations - research then analyze"""
    print("\n=== Example 2: Chained Operations ===")
    
    client = TeloscriptMCPClient("/path/to/teloscript_mcp_server.py")
    
    try:
        await client.connect()
        
        # Step 1: Research task
        research_result = await client.spawn_agent(
            goal="Research Python async programming patterns and save findings to research_async.md",
            mcp_servers=["brave-search", "filesystem"]
        )
        
        if research_result.get("success"):
            research_agent_id = research_result.get("agent_id")
            print(f"✓ Research agent spawned: {research_agent_id}")
            
            # Wait for research to complete (simplified polling)
            for i in range(30):  # Max 30 checks
                status = await client.check_agent_status(research_agent_id)
                if status.get("success") and status.get("status_data", {}).get("status") == "completed":
                    print("✓ Research completed!")
                    break
                await asyncio.sleep(2)
            
            # Step 2: Analysis task
            analysis_result = await client.spawn_agent(
                goal="Read research_async.md and create implementation recommendations in recommendations.md",
                mcp_servers=["filesystem"]
            )
            
            if analysis_result.get("success"):
                analysis_agent_id = analysis_result.get("agent_id")
                print(f"✓ Analysis agent spawned: {analysis_agent_id}")
        
    finally:
        await client.disconnect()


async def example_3_purpose_endpoints():
    """Example 3: Using purpose endpoints"""
    print("\n=== Example 3: Purpose Endpoints ===")
    
    client = TeloscriptMCPClient("/path/to/teloscript_mcp_server.py")
    
    try:
        await client.connect()
        
        # List available purpose endpoints
        endpoints = await client.list_purpose_endpoints()
        print(f"Available endpoints: {endpoints}")
        
        # Execute a purpose endpoint (example)
        if endpoints.get("success") and endpoints.get("endpoints"):
            # Use the first available endpoint as an example
            first_endpoint = endpoints["endpoints"][0]
            endpoint_slug = first_endpoint.get("slug")
            
            if endpoint_slug:
                result = await client.execute_purpose_endpoint(
                    endpoint_slug=endpoint_slug,
                    input_data={"example": "data"}
                )
                print(f"Purpose endpoint result: {result}")
    
    finally:
        await client.disconnect()


async def example_4_agent_management():
    """Example 4: Agent management and monitoring"""
    print("\n=== Example 4: Agent Management ===")
    
    client = TeloscriptMCPClient("/path/to/teloscript_mcp_server.py")
    
    try:
        await client.connect()
        
        # Get system capabilities
        capabilities = await client.get_agent_capabilities()
        print(f"System capabilities: {capabilities}")
        
        # List active agents
        active_agents = await client.list_active_agents()
        print(f"Active agents: {active_agents}")
        
        # Spawn multiple agents for demonstration
        agents = []
        for i in range(3):
            result = await client.spawn_agent(
                goal=f"Task {i+1}: Count files in current directory",
                mcp_servers=["filesystem"],
                max_iterations=5
            )
            if result.get("success"):
                agents.append(result.get("agent_id"))
                print(f"✓ Spawned agent {i+1}: {result.get('agent_id')}")
        
        # Monitor all agents
        print("\nMonitoring agents...")
        for agent_id in agents:
            status = await client.check_agent_status(agent_id)
            print(f"Agent {agent_id}: {status.get('status_data', {}).get('status', 'unknown')}")
        
        # Cancel one agent (for demonstration)
        if agents:
            cancel_result = await client.cancel_agent(agents[0])
            print(f"Cancel result: {cancel_result}")
    
    finally:
        await client.disconnect()


async def example_5_recursive_orchestration():
    """Example 5: Recursive orchestration - agent spawning agents"""
    print("\n=== Example 5: Recursive Orchestration ===")
    
    client = TeloscriptMCPClient("/path/to/teloscript_mcp_server.py")
    
    try:
        await client.connect()
        
        # Master agent that orchestrates sub-agents
        master_goal = """
        You are a master orchestrator. Your task is to coordinate multiple specialized agents:
        
        1. Spawn a research agent to gather information about "Python web frameworks"
        2. Spawn another agent to analyze the research and create a comparison
        3. Spawn a final agent to create a recommendation report
        
        Use the TELOSCRIPT MCP tools available to you to spawn and manage these agents.
        Monitor their progress and ensure all tasks are completed successfully.
        """
        
        master_result = await client.spawn_agent(
            goal=master_goal,
            mcp_servers=["teloscript", "filesystem"],  # Include teloscript as an MCP server!
            max_iterations=20,
            timeout=600
        )
        
        if master_result.get("success"):
            master_agent_id = master_result.get("agent_id")
            print(f"✓ Master orchestrator spawned: {master_agent_id}")
            
            # Monitor the master orchestrator
            print("Monitoring master orchestrator...")
            for i in range(60):  # Max 60 checks (20 minutes)
                status = await client.check_agent_status(master_agent_id)
                if status.get("success"):
                    agent_status = status.get("status_data", {}).get("status")
                    activity = status.get("status_data", {}).get("current_activity", "")
                    print(f"[{i*20}s] Status: {agent_status} - {activity}")
                    
                    if agent_status in ["completed", "failed", "cancelled"]:
                        print(f"✓ Master orchestration {agent_status}!")
                        break
                
                await asyncio.sleep(20)  # Check every 20 seconds
        else:
            print(f"✗ Failed to spawn master orchestrator: {master_result.get('error')}")
    
    finally:
        await client.disconnect()


async def main():
    """Run all examples"""
    print("TELOSCRIPT MCP Server Client Examples")
    print("=====================================")
    
    # Note: Update the path to your actual teloscript_mcp_server.py location
    print("\nIMPORTANT: Update the script path in each example to point to your teloscript_mcp_server.py")
    
    try:
        await example_1_simple_research()
        await example_2_chained_operations()
        await example_3_purpose_endpoints()
        await example_4_agent_management()
        await example_5_recursive_orchestration()
        
        print("\n✓ All examples completed!")
        
    except Exception as e:
        print(f"✗ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure TELOSCRIPT API is running on http://localhost:8000")
        print("2. Update the script path in the examples")
        print("3. Install required dependencies: pip install mcp")


if __name__ == "__main__":
    asyncio.run(main())