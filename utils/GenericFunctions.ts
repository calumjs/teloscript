import {
	IExecuteFunctions,
	ILoadOptionsFunctions,
} from 'n8n-core';
import {
	IDataObject,
	IHttpRequestMethods,
	IRequestOptions,
	NodeApiError,
} from 'n8n-workflow';

/**
 * Make an authenticated API request to TELOSCRIPT
 */
export async function teloscriptApiRequest(
	this: IExecuteFunctions | ILoadOptionsFunctions,
	method: IHttpRequestMethods,
	endpoint: string,
	body: IDataObject = {},
	qs: IDataObject = {},
): Promise<any> {
	const credentials = await this.getCredentials('teloscriptApi');
	
	if (!credentials) {
		throw new NodeApiError(this.getNode(), {
			message: 'No credentials found for TELOSCRIPT API',
		});
	}

	const baseUrl = (credentials.baseUrl as string).replace(/\/$/, '');
	
	const options: IRequestOptions = {
		method,
		body,
		qs,
		url: `${baseUrl}${endpoint}`,
		json: true,
		headers: {
			'Content-Type': 'application/json',
			'Accept': 'application/json',
		},
	};

	// Add authentication if API key is provided
	if (credentials.apiKey) {
		options.headers!['Authorization'] = `Bearer ${credentials.apiKey}`;
	}

	try {
		return await this.helpers.request(options);
	} catch (error) {
		// Enhanced error handling for common TELOSCRIPT API issues
		if (error.response?.body) {
			const errorBody = error.response.body;
			
			// Handle different error response formats
			if (errorBody.detail) {
				throw new NodeApiError(this.getNode(), {
					message: `TELOSCRIPT API Error: ${errorBody.detail}`,
					description: errorBody.description || '',
				});
			} else if (errorBody.error) {
				throw new NodeApiError(this.getNode(), {
					message: `TELOSCRIPT API Error: ${errorBody.error}`,
				});
			}
		}

		// Handle network/connection errors
		if (error.code === 'ECONNREFUSED') {
			throw new NodeApiError(this.getNode(), {
				message: 'Could not connect to TELOSCRIPT API',
				description: 'Please check if your TELOSCRIPT instance is running and the base URL is correct.',
			});
		}

		throw new NodeApiError(this.getNode(), error);
	}
}

/**
 * Poll for agent completion status
 */
export async function pollAgentStatus(
	this: IExecuteFunctions,
	agentId: string,
	maxAttempts: number = 60,
	intervalMs: number = 5000,
): Promise<any> {
	let attempts = 0;
	
	while (attempts < maxAttempts) {
		try {
			const status = await teloscriptApiRequest.call(this, 'GET', `/agents/${agentId}/status`);
			
			if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
				return status;
			}
			
			// Wait before next attempt
			await new Promise(resolve => setTimeout(resolve, intervalMs));
			attempts++;
		} catch (error) {
			// If we can't get status, wait and try again unless we've exceeded max attempts
			if (attempts >= maxAttempts - 1) {
				throw error;
			}
			await new Promise(resolve => setTimeout(resolve, intervalMs));
			attempts++;
		}
	}
	
	throw new NodeApiError(this.getNode(), {
		message: 'Agent execution timeout',
		description: `Agent ${agentId} did not complete within the expected time frame.`,
	});
}

/**
 * Validate and parse JSON input
 */
export function parseJsonInput(input: string, fieldName: string = 'input'): IDataObject {
	if (!input || input.trim() === '') {
		return {};
	}
	
	try {
		return JSON.parse(input);
	} catch (error) {
		throw new Error(`Invalid JSON in ${fieldName}: ${error.message}`);
	}
}

/**
 * Format MCP server configuration from n8n UI format
 */
export function formatMcpServers(serverConfigs: any[]): IDataObject[] {
	if (!Array.isArray(serverConfigs)) {
		return [];
	}
	
	return serverConfigs.map((config: any) => {
		const server: IDataObject = {
			name: config.name || '',
			command: config.command || '',
			transport: config.transport || 'stdio',
		};
		
		// Process arguments
		if (config.args) {
			server.args = config.args.split(',').map((arg: string) => arg.trim()).filter(Boolean);
		}
		
		// Process environment variables
		if (config.env && config.env.envValues && Array.isArray(config.env.envValues)) {
			const envVars: IDataObject = {};
			for (const envVar of config.env.envValues) {
				if (envVar.name && envVar.value) {
					envVars[envVar.name] = envVar.value;
				}
			}
			if (Object.keys(envVars).length > 0) {
				server.env = envVars;
			}
		}
		
		return server;
	});
}

/**
 * Stream response handler for real-time updates
 */
export async function handleStreamResponse(
	this: IExecuteFunctions,
	response: any,
): Promise<any> {
	// For now, return the response as-is
	// In a full implementation, you might want to handle Server-Sent Events
	// or WebSocket connections for real-time streaming
	return response;
}