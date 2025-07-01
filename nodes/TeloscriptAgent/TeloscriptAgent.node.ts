import { IExecuteFunctions } from 'n8n-core';
import {
	IDataObject,
	ILoadOptionsFunctions,
	INodeExecutionData,
	INodePropertyOptions,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { teloscriptApiRequest } from '../utils/GenericFunctions';

export class TeloscriptAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'TELOSCRIPT Agent',
		name: 'teloscriptAgent',
		icon: 'file:teloscript.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
		description: 'Launch and manage TELOSCRIPT autonomous agents',
		defaults: {
			name: 'TELOSCRIPT Agent',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'teloscriptApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Agent',
						value: 'agent',
					},
				],
				default: 'agent',
			},
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['agent'],
					},
				},
				options: [
					{
						name: 'Launch Simple',
						value: 'launchSimple',
						description: 'Launch an agent with a simple text goal',
						action: 'Launch agent with simple goal',
					},
					{
						name: 'Launch Advanced',
						value: 'launchAdvanced',
						description: 'Launch an agent with custom MCP server configuration',
						action: 'Launch agent with advanced configuration',
					},
					{
						name: 'Get Status',
						value: 'getStatus',
						description: 'Get the status of a running agent',
						action: 'Get agent status',
					},
					{
						name: 'Cancel',
						value: 'cancel',
						description: 'Cancel a running agent',
						action: 'Cancel agent',
					},
					{
						name: 'List All',
						value: 'listAll',
						description: 'List all active agents',
						action: 'List all agents',
					},
				],
				default: 'launchSimple',
			},
			// Simple Launch Fields
			{
				displayName: 'Goal',
				name: 'goal',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['launchSimple'],
						resource: ['agent'],
					},
				},
				default: '',
				placeholder: 'Analyze the latest trends in AI and create a summary report',
				description: 'The goal you want the agent to achieve',
			},
			// Advanced Launch Fields
			{
				displayName: 'Goal',
				name: 'goalAdvanced',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['launchAdvanced'],
						resource: ['agent'],
					},
				},
				default: '',
				placeholder: 'Research Python async patterns and save findings to a report',
				description: 'The goal you want the agent to achieve',
			},
			{
				displayName: 'MCP Servers',
				name: 'mcpServers',
				type: 'fixedCollection',
				typeOptions: {
					multipleValues: true,
				},
				displayOptions: {
					show: {
						operation: ['launchAdvanced'],
						resource: ['agent'],
					},
				},
				default: {},
				placeholder: 'Add MCP Server',
				options: [
					{
						name: 'serverValues',
						displayName: 'MCP Server',
						values: [
							{
								displayName: 'Server Name',
								name: 'name',
								type: 'string',
								default: '',
								placeholder: 'filesystem',
								description: 'Name of the MCP server',
							},
							{
								displayName: 'Command',
								name: 'command',
								type: 'string',
								default: '',
								placeholder: 'npx',
								description: 'Command to run the MCP server',
							},
							{
								displayName: 'Arguments',
								name: 'args',
								type: 'string',
								default: '',
								placeholder: '-y,@modelcontextprotocol/server-filesystem,.',
								description: 'Comma-separated arguments for the command',
							},
							{
								displayName: 'Environment Variables',
								name: 'env',
								type: 'fixedCollection',
								typeOptions: {
									multipleValues: true,
								},
								default: {},
								options: [
									{
										name: 'envValues',
										displayName: 'Environment Variable',
										values: [
											{
												displayName: 'Name',
												name: 'name',
												type: 'string',
												default: '',
											},
											{
												displayName: 'Value',
												name: 'value',
												type: 'string',
												default: '',
											},
										],
									},
								],
							},
							{
								displayName: 'Transport',
								name: 'transport',
								type: 'options',
								options: [
									{
										name: 'stdio',
										value: 'stdio',
									},
									{
										name: 'http',
										value: 'http',
									},
								],
								default: 'stdio',
							},
						],
					},
				],
			},
			{
				displayName: 'Additional Options',
				name: 'additionalFields',
				type: 'collection',
				placeholder: 'Add Field',
				default: {},
				displayOptions: {
					show: {
						operation: ['launchSimple', 'launchAdvanced'],
						resource: ['agent'],
					},
				},
				options: [
					{
						displayName: 'Max Iterations',
						name: 'maxIterations',
						type: 'number',
						default: 15,
						description: 'Maximum number of iterations for the agent',
					},
					{
						displayName: 'Timeout (seconds)',
						name: 'timeout',
						type: 'number',
						default: 180,
						description: 'Timeout for the agent execution in seconds',
					},
					{
						displayName: 'Wait for Completion',
						name: 'waitForCompletion',
						type: 'boolean',
						default: true,
						description: 'Whether to wait for the agent to complete before returning',
					},
				],
			},
			// Status/Cancel Operation Fields
			{
				displayName: 'Agent ID',
				name: 'agentId',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['getStatus', 'cancel'],
						resource: ['agent'],
					},
				},
				default: '',
				description: 'The ID of the agent to check or cancel',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const resource = this.getNodeParameter('resource', 0) as string;
		const operation = this.getNodeParameter('operation', 0) as string;

		for (let i = 0; i < items.length; i++) {
			try {
				if (resource === 'agent') {
					if (operation === 'launchSimple') {
						const goal = this.getNodeParameter('goal', i) as string;
						const additionalFields = this.getNodeParameter('additionalFields', i) as IDataObject;

						const body: IDataObject = {
							goal,
							max_iterations: additionalFields.maxIterations || 15,
							timeout: additionalFields.timeout || 180,
						};

						const waitForCompletion = additionalFields.waitForCompletion !== false;

						if (waitForCompletion) {
							// Launch and wait for completion
							const response = await teloscriptApiRequest.call(this, 'POST', '/agents', body);
							returnData.push({
								json: response,
								pairedItem: { item: i },
							});
						} else {
							// Just launch and return agent ID
							const response = await teloscriptApiRequest.call(this, 'POST', '/agents', body);
							returnData.push({
								json: { agentId: response.agent_id, status: 'launched' },
								pairedItem: { item: i },
							});
						}
					} else if (operation === 'launchAdvanced') {
						const goal = this.getNodeParameter('goalAdvanced', i) as string;
						const additionalFields = this.getNodeParameter('additionalFields', i) as IDataObject;
						const mcpServers = this.getNodeParameter('mcpServers', i) as IDataObject;

						// Process MCP servers configuration
						const servers: IDataObject[] = [];
						if (mcpServers.serverValues && Array.isArray(mcpServers.serverValues)) {
							for (const serverConfig of mcpServers.serverValues as IDataObject[]) {
								const server: IDataObject = {
									name: serverConfig.name,
									command: serverConfig.command,
									args: serverConfig.args ? (serverConfig.args as string).split(',').map((arg: string) => arg.trim()) : [],
									transport: serverConfig.transport || 'stdio',
								};

								// Process environment variables
								if (serverConfig.env && (serverConfig.env as IDataObject).envValues) {
									const envVars: IDataObject = {};
									const envValues = (serverConfig.env as IDataObject).envValues as IDataObject[];
									for (const envVar of envValues) {
										envVars[envVar.name as string] = envVar.value;
									}
									if (Object.keys(envVars).length > 0) {
										server.env = envVars;
									}
								}

								servers.push(server);
							}
						}

						const body: IDataObject = {
							goal,
							servers,
							max_iterations: additionalFields.maxIterations || 15,
							timeout: additionalFields.timeout || 180,
						};

						const waitForCompletion = additionalFields.waitForCompletion !== false;

						if (waitForCompletion) {
							const response = await teloscriptApiRequest.call(this, 'POST', '/agents', body);
							returnData.push({
								json: response,
								pairedItem: { item: i },
							});
						} else {
							const response = await teloscriptApiRequest.call(this, 'POST', '/agents', body);
							returnData.push({
								json: { agentId: response.agent_id, status: 'launched' },
								pairedItem: { item: i },
							});
						}
					} else if (operation === 'getStatus') {
						const agentId = this.getNodeParameter('agentId', i) as string;
						const response = await teloscriptApiRequest.call(this, 'GET', `/agents/${agentId}/status`);
						returnData.push({
							json: response,
							pairedItem: { item: i },
						});
					} else if (operation === 'cancel') {
						const agentId = this.getNodeParameter('agentId', i) as string;
						const response = await teloscriptApiRequest.call(this, 'DELETE', `/agents/${agentId}`);
						returnData.push({
							json: { agentId, status: 'cancelled', ...response },
							pairedItem: { item: i },
						});
					} else if (operation === 'listAll') {
						const response = await teloscriptApiRequest.call(this, 'GET', '/agents');
						returnData.push({
							json: response,
							pairedItem: { item: i },
						});
					}
				}
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: { error: error.message },
						pairedItem: { item: i },
					});
				} else {
					throw error;
				}
			}
		}

		return [returnData];
	}
}