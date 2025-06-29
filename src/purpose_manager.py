import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
from datetime import datetime

try:
    from .models import (
        PurposeEndpointConfig, PurposeEndpointRequest, PurposeEndpointResponse,
        PurposeEndpointLearnings, MCPAgentConfig, AgentGoal, MCPServerConfig, 
        AgentRequest, AgentStatus
    )
    from .orchestrator import WorkflowOrchestrator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from models import (
        PurposeEndpointConfig, PurposeEndpointRequest, PurposeEndpointResponse,
        PurposeEndpointLearnings, MCPAgentConfig, AgentGoal, MCPServerConfig, 
        AgentRequest, AgentStatus
    )
    from orchestrator import WorkflowOrchestrator

class PurposeEndpointManager:
    """Manages purpose endpoint configurations and execution"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.endpoints_file = self.config_dir / "purpose_endpoints.json"
        self.endpoints: Dict[str, PurposeEndpointConfig] = {}
        self.orchestrator = WorkflowOrchestrator()
        self.load_endpoints()
    
    def load_endpoints(self) -> None:
        """Load purpose endpoints from configuration file"""
        try:
            self.config_dir.mkdir(exist_ok=True)
            
            if self.endpoints_file.exists():
                with open(self.endpoints_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.endpoints = {
                        slug: PurposeEndpointConfig(**config)
                        for slug, config in data.items()
                    }
                logger.info(f"Loaded {len(self.endpoints)} purpose endpoints")
            else:
                self._create_default_endpoints()
                self.save_endpoints()
                
        except Exception as e:
            logger.error(f"Error loading purpose endpoints: {e}")
            self.endpoints = {}
    
    def save_endpoints(self) -> bool:
        """Save purpose endpoints to configuration file"""
        try:
            data = {
                slug: endpoint.dict()
                for slug, endpoint in self.endpoints.items()
            }
            
            with open(self.endpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.endpoints)} purpose endpoints")
            return True
            
        except Exception as e:
            logger.error(f"Error saving purpose endpoints: {e}")
            return False
    
    def _create_default_endpoints(self) -> None:
        """Create default purpose endpoints"""
        default_endpoints = {
            "handle-github-webhook": PurposeEndpointConfig(
                slug="handle-github-webhook",
                name="GitHub Webhook Handler",
                description="Consume GitHub webhooks and process them appropriately",
                prompt_template="Consume this webhook data and process it as appropriate. Analyze the webhook type, extract relevant information, and take appropriate actions based on the event type and payload. Consider the repository context, user permissions, and event-specific requirements.",
                mcp_servers=["github"],
                max_iterations=15,
                timeout=300,
                tags=["github", "webhook", "automation"]
            ),
            "analyze-code-changes": PurposeEndpointConfig(
                slug="analyze-code-changes",
                name="Code Change Analyzer",
                description="Analyze code changes and provide insights",
                prompt_template="Analyze the provided code changes thoroughly. Identify the type of changes, potential impacts on the codebase, code quality issues, security concerns, and provide specific recommendations for improvement. Consider best practices, maintainability, and potential risks.",
                mcp_servers=["filesystem", "github"],
                max_iterations=12,
                timeout=240,
                tags=["code-analysis", "review", "quality"]
            ),
            "research-topic": PurposeEndpointConfig(
                slug="research-topic",
                name="Topic Researcher",
                description="Research a given topic using web search and document findings",
                prompt_template="Research the provided topic thoroughly using available resources. Search for current information, gather relevant data from multiple sources, analyze trends and patterns, and compile a comprehensive report with proper citations, insights, and actionable conclusions.",
                mcp_servers=["brave-search", "filesystem"],
                max_iterations=20,
                timeout=600,
                tags=["research", "information-gathering", "analysis"]
            )
        }
        
        self.endpoints.update(default_endpoints)
        logger.info("Created default purpose endpoints")
    
    def get_endpoint(self, slug: str) -> Optional[PurposeEndpointConfig]:
        """Get a purpose endpoint by slug"""
        return self.endpoints.get(slug)
    
    def list_endpoints(self) -> List[PurposeEndpointConfig]:
        """List all purpose endpoints"""
        return list(self.endpoints.values())
    
    def create_endpoint(self, config: PurposeEndpointConfig) -> bool:
        """Create a new purpose endpoint"""
        try:
            if config.slug in self.endpoints:
                raise ValueError(f"Endpoint with slug '{config.slug}' already exists")
            
            self.endpoints[config.slug] = config
            self.save_endpoints()
            logger.info(f"Created purpose endpoint: {config.slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating purpose endpoint: {e}")
            return False
    
    def update_endpoint(self, slug: str, config: PurposeEndpointConfig) -> bool:
        """Update an existing purpose endpoint"""
        try:
            if slug not in self.endpoints:
                raise ValueError(f"Endpoint with slug '{slug}' not found")
            
            config.slug = slug  # Ensure slug matches
            config.updated_at = datetime.utcnow()
            self.endpoints[slug] = config
            self.save_endpoints()
            logger.info(f"Updated purpose endpoint: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating purpose endpoint: {e}")
            return False
    
    def delete_endpoint(self, slug: str) -> bool:
        """Delete a purpose endpoint"""
        try:
            if slug not in self.endpoints:
                raise ValueError(f"Endpoint with slug '{slug}' not found")
            
            del self.endpoints[slug]
            self.save_endpoints()
            logger.info(f"Deleted purpose endpoint: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting purpose endpoint: {e}")
            return False
    
    async def execute_endpoint(self, request: PurposeEndpointRequest, status_callback=None) -> PurposeEndpointResponse:
        """Execute a purpose endpoint"""
        try:
            # Get the endpoint configuration
            endpoint = self.get_endpoint(request.endpoint_slug)
            if not endpoint:
                raise ValueError(f"Purpose endpoint '{request.endpoint_slug}' not found")
            
            if not endpoint.enabled:
                raise ValueError(f"Purpose endpoint '{request.endpoint_slug}' is disabled")
            
            # Build the prompt with input data
            prompt = self._build_prompt(endpoint, request.input_data)
            
            # Create MCP server configurations
            mcp_configs = self._get_mcp_configs(endpoint.mcp_servers)
            
            # Create agent configuration
            # Convert input_data to dict for context, or use empty dict
            context = {}
            if isinstance(request.input_data, dict):
                context = request.input_data
            elif request.input_data is not None:
                context = {"input": request.input_data}
            
            agent_config = MCPAgentConfig(
                servers=mcp_configs,
                goal=AgentGoal(
                    description=prompt,
                    context=context
                ),
                max_iterations=endpoint.max_iterations,
                timeout=endpoint.timeout
            )
            
            # Create agent request
            agent_request = AgentRequest(
                id=request.request_id,
                config=agent_config
            )
            
            # Execute the agent
            start_time = time.time()
            agent_response = await self.orchestrator.run_single_agent(
                agent_request, 
                status_callback
            )
            
            execution_time = time.time() - start_time
            
            # Build response
            response = PurposeEndpointResponse(
                request_id=request.request_id,
                endpoint_slug=request.endpoint_slug,
                status=agent_response.status,
                result=agent_response.result,
                execution_time=execution_time,
                iterations_used=agent_response.iterations_used,
                input_data=request.input_data,
                error=agent_response.error,
                completed_at=datetime.utcnow()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing purpose endpoint: {e}")
            return PurposeEndpointResponse(
                request_id=request.request_id,
                endpoint_slug=request.endpoint_slug,
                status=AgentStatus.FAILED,
                result="",
                execution_time=0,
                iterations_used=0,
                input_data=request.input_data,
                error=str(e),
                completed_at=datetime.utcnow()
            )
    
    def _build_prompt(self, endpoint: PurposeEndpointConfig, input_data: Any) -> str:
        """Build the final prompt by combining template with input data"""
        # Simple template substitution for now
        # Could be enhanced with more sophisticated templating
        prompt = endpoint.prompt_template
        
        # Convert input data to string format
        if input_data:
            if isinstance(input_data, str):
                input_str = input_data
            else:
                # Convert any other type to JSON string
                try:
                    input_str = json.dumps(input_data, indent=2)
                except (TypeError, ValueError):
                    # If JSON serialization fails, convert to string
                    input_str = str(input_data)
            
            prompt += f"\n\nInput Data:\n{input_str}"
        
        return prompt
    
    def _get_mcp_configs(self, server_names: List[str]) -> List[MCPServerConfig]:
        """Get MCP server configurations by name"""
        # Import here to avoid circular imports
        try:
            from .api import mcp_configs
        except ImportError:
            # Fallback: load configs directly
            mcp_configs = self._load_mcp_configs_direct()
        
        server_configs = []
        for server_name in server_names:
            if server_name in mcp_configs:
                config_data = mcp_configs[server_name]["config"]
                server_configs.append(MCPServerConfig(**config_data))
            else:
                logger.warning(f"MCP server '{server_name}' not found in configuration")
        
        return server_configs
    
    def _load_mcp_configs_direct(self) -> Dict[str, Dict]:
        """Load MCP configs directly from file (fallback method)"""
        try:
            mcp_config_file = self.config_dir / "mcp_configs.json"
            if mcp_config_file.exists():
                with open(mcp_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading MCP configs directly: {e}")
        
        return {} 