import asyncio
import time
import uuid
import json
import os
import sys
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Fix imports
try:
    from .models import (
        MCPAgentConfig, AgentGoal, AgentStatus, StatusUpdate, 
        AgentResponse, MCPServerConnection, ToolInvocation,
        ResourceRequest, PromptInvocation, MCPToolDefinition
    )
    from .mcp_host import MCPHost, MCPProtocolError
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from models import (
        MCPAgentConfig, AgentGoal, AgentStatus, StatusUpdate, 
        AgentResponse, MCPServerConnection, ToolInvocation,
        ResourceRequest, PromptInvocation, MCPToolDefinition
    )
    from mcp_host import MCPHost, MCPProtocolError

class AgentStepResponse(BaseModel):
    """Structured response for each agent iteration"""
    complete: bool = Field(description="True if the goal has been fully accomplished and no more work is needed")
    should_stop: bool = Field(description="True if there's no point in continuing due to errors, blockers, or impossibility")
    notes: str = Field(description="Brief explanation of what was accomplished or what still needs to be done")
    final_answer: Optional[str] = Field(default=None, description="The complete answer to the user's question (only if complete=True)")

class LLMPoweredMCPAgent:
    """MCP Agent that executes goals using GPT-4.1 for intelligent planning and execution"""
    
    def __init__(self, agent_id: str, config: MCPAgentConfig, status_callback: Optional[Callable] = None):
        self.agent_id = agent_id
        self.config = config
        self.status_callback = status_callback
        self.host = MCPHost()
        self.status = AgentStatus.PENDING
        self.status_updates: List[StatusUpdate] = []
        self.start_time = time.time()
        self.iterations_used = 0
        self.cancelled = False
        
        # Initialize OpenAI client for GPT-4.1
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm_client = AsyncOpenAI(api_key=api_key)
        self.conversation_history = []
        self.tool_schemas = {}  # Will store tool schemas for LLM
        
    async def _emit_status(self, status: AgentStatus, message: str, progress: Optional[float] = None, details: Optional[Dict] = None):
        """Emit status update"""
        self.status = status
        update = StatusUpdate(
            agent_id=self.agent_id,
            status=status,
            message=message,
            progress=progress,
            details=details or {}
        )
        self.status_updates.append(update)
        
        if self.status_callback:
            try:
                await self.status_callback(update)
            except Exception as e:
                logger.warning(f"Status callback error: {e}")
        
        logger.info(f"Agent {self.agent_id}: {message}")
    
    async def run(self) -> AgentResponse:
        """Execute the agent's goal using GPT-4.1 intelligence"""
        try:
            await self._emit_status(AgentStatus.RUNNING, "Initializing LLM-powered agent")
            
            # Connect to MCP servers
            await self._emit_status(AgentStatus.RUNNING, "Connecting to MCP servers", 10)
            server_connections = await self.host.connect_to_servers(self.config.servers)
            
            # Check for connection failures
            connected_servers = [conn for conn in server_connections if conn.connected]
            failed_servers = [conn for conn in server_connections if not conn.connected]
            
            if failed_servers:
                error_msg = f"Failed to connect to servers: {[s.name for s in failed_servers]}"
                logger.warning(error_msg)
            
            if not connected_servers:
                raise Exception("No MCP servers connected successfully")
            
            await self._emit_status(
                AgentStatus.RUNNING, 
                f"Connected to {len(connected_servers)} servers", 
                20,
                {"connected_servers": [s.name for s in connected_servers]}
            )
            
            # Discover and prepare capabilities for LLM
            await self._emit_status(AgentStatus.RUNNING, "Analyzing available capabilities", 30)
            capabilities = await self._prepare_capabilities_for_llm()
            
            await self._emit_status(
                AgentStatus.RUNNING,
                f"Prepared {capabilities['tool_count']} tools, {capabilities['resource_count']} resources, {capabilities['prompt_count']} prompts for LLM",
                40,
                capabilities['summary']
            )
            
            # Execute LLM-powered goal workflow
            result = await self._execute_llm_workflow(capabilities)
            
            # Generate final response
            execution_time = time.time() - self.start_time
            
            # Set status to completed
            self.status = AgentStatus.COMPLETED
            logger.info(f"Agent {self.agent_id} status set to: {self.status} (type: {type(self.status)})")
            
            response = AgentResponse(
                agent_id=self.agent_id,
                status=self.status,
                result=result,
                execution_time=execution_time,
                iterations_used=self.iterations_used,
                status_updates=self.status_updates,
                server_connections=server_connections
            )
            
            logger.info(f"Agent {self.agent_id} response status: {response.status} (type: {type(response.status)})")
            
            await self._emit_status(
                AgentStatus.COMPLETED, 
                f"Goal completed: {result[:200]}...", 
                100,
                {
                    "iterations_used": self.iterations_used,
                    "result": result,
                    "execution_time": execution_time
                }
            )
            return response
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            await self._emit_status(AgentStatus.FAILED, error_msg)
            
            execution_time = time.time() - self.start_time
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                result="",
                execution_time=execution_time,
                iterations_used=self.iterations_used,
                status_updates=self.status_updates,
                server_connections=await self.host.connect_to_servers([]) if hasattr(self, 'host') else [],
                error=error_msg
            )
        finally:
            # Clean up connections
            try:
                await self.host.disconnect_all()
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
    
    async def _prepare_capabilities_for_llm(self) -> Dict[str, Any]:
        """Prepare available MCP capabilities in a format suitable for LLM planning"""
        all_tools = self.host.get_all_tools()
        all_resources = self.host.get_all_resources()
        all_prompts = self.host.get_all_prompts()
        
        # Convert tools to OpenAI function calling format
        tool_functions = []
        for server_name, tools in all_tools.items():
            for tool in tools:
                function_def = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}_{tool.name}",
                        "description": f"[{server_name}] {tool.description}",
                        "parameters": tool.inputSchema
                    }
                }
                tool_functions.append(function_def)
                
                # Store mapping for execution
                self.tool_schemas[f"{server_name}_{tool.name}"] = {
                    "server": server_name,
                    "tool": tool.name,
                    "schema": tool.inputSchema
                }
        
        # Prepare resource and prompt information for context
        resource_info = []
        for server_name, resources in all_resources.items():
            for resource in resources:
                resource_info.append({
                    "server": server_name,
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description
                })
        
        prompt_info = []
        for server_name, prompts in all_prompts.items():
            for prompt in prompts:
                prompt_info.append({
                    "server": server_name,
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": prompt.arguments
                })
        
        return {
            "tool_functions": tool_functions,
            "resources": resource_info,
            "prompts": prompt_info,
            "tool_count": len(tool_functions),
            "resource_count": len(resource_info),
            "prompt_count": len(prompt_info),
            "summary": {
                "tools_by_server": {server: len(tools) for server, tools in all_tools.items()},
                "resources_by_server": {server: len(resources) for server, resources in all_resources.items()},
                "prompts_by_server": {server: len(prompts) for server, prompts in all_prompts.items()}
            }
        }
    
    async def _execute_llm_workflow(self, capabilities: Dict[str, Any]) -> str:
        """Execute goal-oriented workflow using GPT-4.1 for planning and decision making"""
        goal = self.config.goal
        max_iterations = self.config.max_iterations
        
        # Initialize conversation with system prompt
        system_prompt = self._create_system_prompt(goal, capabilities)
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        
        # Add user's goal
        user_goal = f"""
Goal: {goal.description}

Success Criteria: {goal.success_criteria or "Complete the goal to the best of your ability"}

Context: {json.dumps(goal.context, indent=2) if goal.context else "No additional context provided"}

Please analyze this goal and create an execution plan using the available MCP tools and resources. 
Start by planning your approach, then execute it step by step.
"""
        
        self.conversation_history.append({"role": "user", "content": user_goal})
        
        await self._emit_status(
            AgentStatus.RUNNING, 
            "GPT-4.1 analyzing goal and creating execution plan", 
            50
        )
        
        # Execute iterative workflow with LLM
        workflow_results = []
        
        for iteration in range(max_iterations):
            if self.cancelled:
                break
                
            self.iterations_used += 1
            
            await self._emit_status(
                AgentStatus.RUNNING,
                f"LLM iteration {iteration + 1}/{max_iterations}",
                progress=50 + (iteration / max_iterations) * 40
            )
            
            try:
                # Get LLM response with potential tool calls
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4.1",  # Actual GPT-4.1 model
                    messages=self.conversation_history,
                    tools=capabilities["tool_functions"] if capabilities["tool_functions"] else None,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=2000
                )
                
                message = response.choices[0].message
                self.conversation_history.append(message.model_dump(exclude_unset=True))
                
                # Handle tool calls if any
                if message.tool_calls:
                    tool_results = await self._execute_tool_calls(message.tool_calls)
                    workflow_results.extend(tool_results)
                    
                    # Add tool results to conversation
                    for tool_call, result in zip(message.tool_calls, tool_results):
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result["result"])
                        })
                
                # ALWAYS run structured completion assessment regardless of tool calls
                # This ensures consistent reasoning emission and termination logic every iteration
                
                # Prepare completion prompt
                completion_prompt = f"""
Based on the current state above, evaluate the current state of this goal: "{goal.description}"

EVALUATION CRITERIA:
1. **COMPLETE**: Have you gathered all information needed to fully answer the user's question? If yes, provide the complete answer.

2. **SHOULD STOP**: Is there no point in continuing? This includes:
   - Encountered unrecoverable errors (authentication failures, permission denied, resource not found)
   - Hit a technical blocker that cannot be resolved with available tools
   - Task is impossible or misconfigured
   - Already failed multiple attempts at the same approach
   - Tool keeps returning errors that indicate fundamental issues

3. **CONTINUE**: If not complete and there's still a viable path forward, explain what needs to be done next.

Respond with a structured assessment:
- complete: true if you have all information needed to fully answer the user's question
- should_stop: true if there's no point in continuing due to errors, blockers, or impossibility  
- notes: brief explanation of what you accomplished or what still needs to be done
- final_answer: if complete=true, provide the complete answer to the user's question

Be decisive and realistic: Don't continue if you're stuck in error loops or hitting fundamental blockers.
"""
                    
                self.conversation_history.append({
                    "role": "user", 
                    "content": completion_prompt
                })
                
                # Get structured response about completion
                completion_response = await self.llm_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=self.conversation_history,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "completion_assessment",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "complete": {
                                        "type": "boolean",
                                        "description": "True if the goal has been fully accomplished and no more work is needed"
                                    },
                                    "should_stop": {
                                        "type": "boolean",
                                        "description": "True if there's no point in continuing due to errors, blockers, or impossibility"
                                    },
                                    "notes": {
                                        "type": "string", 
                                        "description": "Brief explanation of what was accomplished or what still needs to be done"
                                    },
                                    "final_answer": {
                                        "type": "string",
                                        "description": "The complete answer to the user's question (only if complete=True)"
                                    }
                                },
                                "required": ["complete", "should_stop", "notes"],
                                "additionalProperties": false
                            }
                        }
                    },
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # Parse the structured response with detailed logging
                raw_response = completion_response.choices[0].message.content
                logger.info(f"Raw structured response: {raw_response}")
                
                # Always emit a status update, even if parsing fails
                completion_status_emitted = False
                
                try:
                    step_result_json = json.loads(raw_response)
                    logger.info(f"Parsed JSON: {step_result_json}")
                    
                    step_result = AgentStepResponse(
                        complete=step_result_json.get("complete", False),
                        should_stop=step_result_json.get("should_stop", False),
                        notes=step_result_json.get("notes", ""),
                        final_answer=step_result_json.get("final_answer")
                    )
                    
                    logger.info(f"Structured completion check: complete={step_result.complete}, should_stop={step_result.should_stop}, notes='{step_result.notes}', final_answer='{step_result.final_answer}'")
                    
                    # Emit the agent's reasoning as a status update
                    await self._emit_status(
                        AgentStatus.RUNNING,
                        f"Agent reasoning: {step_result.notes}",
                        progress=50 + (iteration / max_iterations) * 40,
                        details={
                            "iteration": iteration + 1,
                            "complete": step_result.complete,
                            "should_stop": step_result.should_stop,
                            "notes": step_result.notes
                        }
                    )
                    completion_status_emitted = True
                    
                    # Add the structured response to conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": raw_response
                    })
                    
                    if step_result.complete:
                        # Goal is complete, use the final answer
                        logger.info(f"✅ GOAL MARKED COMPLETE - Breaking out of iteration loop at iteration {iteration + 1}")
                        await self._emit_status(
                            AgentStatus.RUNNING,
                            f"Goal completed: {step_result.notes}",
                            progress=90
                        )
                        
                        # Store the final answer for the summary
                        workflow_results.append({
                            "type": "completion",
                            "final_answer": step_result.final_answer or step_result.notes,
                            "iteration": iteration + 1
                        })
                        break
                    elif step_result.should_stop:
                        # Hit a blocker or error - stop early
                        logger.info(f"🛑 EARLY TERMINATION - Stopping at iteration {iteration + 1}: {step_result.notes}")
                        await self._emit_status(
                            AgentStatus.RUNNING,
                            f"Early termination: {step_result.notes}",
                            progress=50
                        )
                        
                        # Store the reason for early termination
                        workflow_results.append({
                            "type": "early_termination",
                            "reason": step_result.notes,
                            "iteration": iteration + 1
                        })
                        break
                    else:
                        # Continue working - log what still needs to be done
                        logger.info(f"❌ GOAL NOT COMPLETE - Continuing iteration {iteration + 1}: {step_result.notes}")
                        workflow_results.append({
                            "type": "progress",
                            "notes": step_result.notes,
                            "iteration": iteration + 1
                        })
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse structured response as JSON: {e}")
                    logger.error(f"Raw response was: {repr(raw_response)}")
                    
                    # Emit error status even when parsing fails
                    if not completion_status_emitted:
                        await self._emit_status(
                            AgentStatus.RUNNING,
                            f"Agent reasoning: JSON parsing error - {str(e)}",
                            progress=50 + (iteration / max_iterations) * 40,
                            details={
                                "iteration": iteration + 1,
                                "error": f"JSON parse error: {str(e)}",
                                "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
                            }
                        )
                    
                    # Fallback: assume not complete and continue
                    workflow_results.append({
                        "type": "error",
                        "error": f"JSON parse error: {str(e)}",
                        "iteration": iteration + 1
                    })
                except Exception as e:
                    logger.error(f"Error processing structured completion response: {e}")
                    logger.error(f"Raw response was: {repr(raw_response)}")
                    
                    # Emit error status even when processing fails
                    if not completion_status_emitted:
                        await self._emit_status(
                            AgentStatus.RUNNING,
                            f"Agent reasoning: Processing error - {str(e)}",
                            progress=50 + (iteration / max_iterations) * 40,
                            details={
                                "iteration": iteration + 1,
                                "error": f"Completion processing error: {str(e)}",
                                "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
                            }
                        )
                    
                    # Fallback: assume not complete and continue
                    workflow_results.append({
                        "type": "error", 
                        "error": f"Completion processing error: {str(e)}",
                        "iteration": iteration + 1
                    })
                
                else:
                    # No tool calls, LLM is providing analysis or conclusion
                    if message.content:
                        workflow_results.append({
                            "type": "analysis",
                            "content": message.content,
                            "iteration": iteration + 1
                        })
                        
                        # For responses without tool calls, also check for completion
                        content_lower = message.content.lower()
                        completion_phrases = ["goal completed", "task completed", "analysis complete", "goal achieved", "task accomplished"]
                        if any(phrase in content_lower for phrase in completion_phrases):
                            await self._emit_status(
                                AgentStatus.RUNNING,
                                "LLM indicates goal completion",
                                progress=90
                            )
                            break
                
            except Exception as e:
                logger.warning(f"LLM iteration {iteration + 1} failed: {e}")
                workflow_results.append({
                    "type": "error",
                    "error": str(e),
                    "iteration": iteration + 1
                })
                
                # Add error to conversation for LLM to handle
                self.conversation_history.append({
                    "role": "user",
                    "content": f"An error occurred: {str(e)}. Please adjust your approach and continue."
                })
        
        # Generate final result summary using LLM
        return await self._generate_llm_summary(goal, workflow_results)
    
    def _create_system_prompt(self, goal: AgentGoal, capabilities: Dict[str, Any]) -> str:
        """Create a comprehensive system prompt for GPT-4.1"""
        return f"""You are an intelligent MCP (Model Context Protocol) agent executor powered by GPT-4.1. Your role is to achieve user goals by intelligently planning and executing tasks using available MCP servers and their capabilities.

AVAILABLE CAPABILITIES:
- {capabilities['tool_count']} Tools available for execution
- {capabilities['resource_count']} Resources available for reading  
- {capabilities['prompt_count']} Prompts available for invocation

TOOLS BY SERVER:
{json.dumps(capabilities['summary']['tools_by_server'], indent=2)}

CORE EXECUTION PRINCIPLES:
1. **UNDERSTAND THE GOAL**: Carefully analyze what the user is asking for and what success looks like
2. **GATHER INFORMATION**: Use available tools to collect the necessary data
3. **PROVIDE FINAL ANSWERS**: Once you have the information, give the complete answer immediately
4. **NO ENDLESS PLANNING**: Don't keep saying "let me" or "I will" - just do the work and provide results
5. **BE DIRECT**: When you have the data needed to answer the question, answer it directly

EXECUTION WORKFLOW:
- Use tools to gather the required information
- Once you have sufficient data, provide the final answer immediately  
- Don't overthink or over-plan - be direct and conclusive
- If you need more information, get it quickly and then conclude

COMPLETION CRITERIA:
- As soon as you have enough information to answer the user's question, provide the final answer
- Use "FINAL ANSWER:" to clearly indicate when you're providing the complete response
- Don't continue iterating if you already have what the user requested

TOOL CALLING BEST PRACTICES:
- Tools are named as "servername_toolname" format
- Each tool has specific input schemas - follow them precisely
- Read tool descriptions carefully to understand their purpose and parameters
- Use appropriate tools for the task at hand
- Chain tool calls logically based on results

QUALITY STANDARDS:
- Provide specific, actionable information rather than vague descriptions
- Include concrete examples, data, and evidence when available
- If you encounter errors, explain what went wrong and try alternative approaches
- Be transparent about limitations or gaps in available information
- Structure your responses clearly and logically

GOAL COMPLETION:
- Only consider a goal complete when you've provided comprehensive, useful information
- If the goal cannot be completed, explain exactly what you tried and why it didn't work
- Always aim to provide maximum value to the user based on available capabilities

ADAPTIVE BEHAVIOR:
- Adjust your strategy based on the types of servers and tools available
- For search servers: Focus on finding and synthesizing relevant information
- For filesystem servers: Explore structure and content systematically  
- For API servers: Make appropriate calls and interpret responses
- For database servers: Query effectively and present results clearly
- For any server type: Understand the capabilities and use them optimally

Remember: You are an autonomous agent capable of working with any type of MCP server. Plan intelligently, execute systematically, and provide comprehensive, useful results regardless of the domain or server type."""

    async def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Execute the tool calls requested by the LLM"""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments JSON: {e}")
                logger.error(f"Raw arguments were: {tool_call.function.arguments}")
                results.append({
                    "tool_call_id": tool_call.id,
                    "server": "unknown",
                    "tool": function_name,
                    "arguments": {},
                    "result": f"Error: Invalid JSON in tool arguments: {str(e)}",
                    "success": False
                })
                continue
            
            try:
                # Look up the actual server and tool name
                if function_name not in self.tool_schemas:
                    raise ValueError(f"Unknown tool: {function_name}")
                
                tool_info = self.tool_schemas[function_name]
                server_name = tool_info["server"]
                tool_name = tool_info["tool"]
                
                await self._emit_status(
                    AgentStatus.RUNNING,
                    f"Executing tool: {server_name}.{tool_name}",
                    details={"arguments": arguments}
                )
                
                # Execute the tool via MCP host
                result = await self.host.call_tool(server_name, tool_name, arguments)
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "server": server_name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                logger.warning(f"Tool call {function_name} failed: {e}")
                results.append({
                    "tool_call_id": tool_call.id,
                    "server": tool_info.get("server", "unknown") if function_name in self.tool_schemas else "unknown",
                    "tool": tool_info.get("tool", function_name) if function_name in self.tool_schemas else function_name,
                    "arguments": arguments,
                    "result": f"Error: {str(e)}",
                    "success": False
                })
        
        return results
    
    async def _generate_llm_summary(self, goal: AgentGoal, workflow_results: List[Dict]) -> str:
        """Generate a final summary using LLM analysis or structured completion result"""
        
        # Check if we have a completion result with final_answer
        completion_results = [r for r in workflow_results if r.get("type") == "completion"]
        if completion_results:
            final_answer = completion_results[-1].get("final_answer")
            if final_answer:
                logger.info("Using final_answer from structured completion")
                return final_answer
        
        # Check if we had an early termination
        termination_results = [r for r in workflow_results if r.get("type") == "early_termination"]
        if termination_results:
            termination_reason = termination_results[-1].get("reason")
            logger.info("Early termination detected, generating appropriate response")
            return f"Unable to complete the goal: {termination_reason}\n\nThe agent encountered issues that prevented successful completion of the requested task."
        
        # Fallback to LLM summary generation
        summary_prompt = f"""
Based on the work completed above, please provide a direct answer to the original user's question/goal:

"{goal.description}"

Provide your answer based on what you discovered and accomplished during the execution. Be direct and informative, answering the user's question as if they asked it directly.

Do not provide a meta-analysis of the workflow or execution process. Simply answer their original question based on the information you gathered.
"""
        
        try:
            # Add summary request to conversation
            self.conversation_history.append({"role": "user", "content": summary_prompt})
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4.1",
                messages=self.conversation_history,
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content or "Summary generation failed"
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            
            # Fallback to simple summary
            successful_tools = len([r for r in workflow_results if r.get("success", False)])
            total_actions = len(workflow_results)
            
            return f"""Goal: {goal.description}

Execution Summary:
- Total actions attempted: {total_actions}
- Successful tool executions: {successful_tools}
- Iterations used: {self.iterations_used}

Results: {'Goal execution completed with mixed results' if successful_tools > 0 else 'Goal execution encountered difficulties'}

Note: Detailed summary generation failed due to: {str(e)}"""
    
    async def cancel(self):
        """Cancel the agent execution"""
        self.cancelled = True
        await self._emit_status(AgentStatus.CANCELLED, "Agent execution cancelled by user")
        await self.host.disconnect_all()

    async def _check_goal_completion(self, goal: AgentGoal) -> str:
        """Check if the LLM believes it has enough information to complete the goal"""
        completion_check_prompt = f"""
Looking at the conversation above, have you successfully completed this goal: "{goal.description}"?

Consider these criteria:
- Have you provided the specific information requested?
- Have you answered the user's question completely?
- Is your response actionable and helpful?
- Would the user be satisfied with what you've delivered?

Respond with EXACTLY one of these formats:
- "READY: [brief explanation of what you accomplished]" - if you have fully satisfied the goal
- "CONTINUE: [what specific work still needs to be done]" - if more work is required

Be honest and precise. Only say READY if you've actually delivered what was requested, not just identified what needs to be done.
"""
        
        try:
            logger.info(f"Checking goal completion for: '{goal.description}'")
            
            # Create a temporary conversation for the completion check
            check_messages = self.conversation_history + [
                {"role": "user", "content": completion_check_prompt}
            ]
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4.1",
                messages=check_messages,
                temperature=0.1,
                max_tokens=200
            )
            
            completion_response = response.choices[0].message.content or "CONTINUE: Unable to determine completion status"
            
            logger.info(f"LLM completion check response: {completion_response}")
            
            # Don't add this to the main conversation history - it's just a check
            return completion_response.strip()
            
        except Exception as e:
            logger.warning(f"Completion check failed: {e}")
            return "CONTINUE: Completion check failed, continuing execution"

# Maintain backward compatibility
MCPAgent = LLMPoweredMCPAgent 