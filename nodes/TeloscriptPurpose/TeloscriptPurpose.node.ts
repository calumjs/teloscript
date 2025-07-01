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

export class TeloscriptPurpose implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'TELOSCRIPT Purpose',
		name: 'teloscriptPurpose',
		icon: 'file:teloscript.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["purposeSlug"]}}',
		description: 'Execute predefined TELOSCRIPT purpose endpoints',
		defaults: {
			name: 'TELOSCRIPT Purpose',
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
						name: 'Purpose',
						value: 'purpose',
					},
				],
				default: 'purpose',
			},
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['purpose'],
					},
				},
				options: [
					{
						name: 'Execute',
						value: 'execute',
						description: 'Execute a purpose endpoint',
						action: 'Execute purpose endpoint',
					},
					{
						name: 'List Available',
						value: 'listAvailable',
						description: 'List all available purpose endpoints',
						action: 'List available purposes',
					},
					{
						name: 'Get Details',
						value: 'getDetails',
						description: 'Get details of a specific purpose endpoint',
						action: 'Get purpose details',
					},
				],
				default: 'execute',
			},
			{
				displayName: 'Purpose Endpoint',
				name: 'purposeSlug',
				type: 'options',
				typeOptions: {
					loadOptionsMethod: 'getPurposeEndpoints',
				},
				displayOptions: {
					show: {
						operation: ['execute', 'getDetails'],
						resource: ['purpose'],
					},
				},
				default: '',
				description: 'The purpose endpoint to execute',
			},
			{
				displayName: 'Input Data',
				name: 'inputData',
				type: 'json',
				displayOptions: {
					show: {
						operation: ['execute'],
						resource: ['purpose'],
					},
				},
				default: '{}',
				description: 'Input data for the purpose endpoint (JSON format)',
			},
			{
				displayName: 'Additional Options',
				name: 'additionalFields',
				type: 'collection',
				placeholder: 'Add Field',
				default: {},
				displayOptions: {
					show: {
						operation: ['execute'],
						resource: ['purpose'],
					},
				},
				options: [
					{
						displayName: 'Wait for Completion',
						name: 'waitForCompletion',
						type: 'boolean',
						default: true,
						description: 'Whether to wait for the purpose execution to complete',
					},
					{
						displayName: 'Stream Response',
						name: 'streamResponse',
						type: 'boolean',
						default: false,
						description: 'Whether to stream the response (for real-time updates)',
					},
				],
			},
		],
	};

	methods = {
		loadOptions: {
			async getPurposeEndpoints(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
				try {
					const purposes = await teloscriptApiRequest.call(this, 'GET', '/purpose/endpoints');
					
					if (Array.isArray(purposes)) {
						return purposes.map((purpose: IDataObject) => ({
							name: `${purpose.name} (${purpose.slug})`,
							value: purpose.slug as string,
							description: purpose.description as string,
						}));
					}
					
					return [];
				} catch (error) {
					console.error('Failed to load purpose endpoints:', error);
					return [];
				}
			},
		},
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const resource = this.getNodeParameter('resource', 0) as string;
		const operation = this.getNodeParameter('operation', 0) as string;

		for (let i = 0; i < items.length; i++) {
			try {
				if (resource === 'purpose') {
					if (operation === 'execute') {
						const purposeSlug = this.getNodeParameter('purposeSlug', i) as string;
						const inputDataParam = this.getNodeParameter('inputData', i) as string;
						const additionalFields = this.getNodeParameter('additionalFields', i) as IDataObject;

						// Parse input data
						let inputData: IDataObject = {};
						try {
							if (inputDataParam) {
								inputData = JSON.parse(inputDataParam);
							}
						} catch (parseError) {
							throw new Error(`Invalid JSON in input data: ${parseError.message}`);
						}

						// Merge with data from previous node if available
						if (items[i].json && Object.keys(items[i].json).length > 0) {
							inputData = { ...items[i].json, ...inputData };
						}

						const body: IDataObject = {
							endpoint_slug: purposeSlug,
							input_data: inputData,
						};

						const waitForCompletion = additionalFields.waitForCompletion !== false;
						const streamResponse = additionalFields.streamResponse === true;

						if (streamResponse) {
							// Use streaming endpoint
							const response = await teloscriptApiRequest.call(this, 'POST', `/purpose/${purposeSlug}/stream`, body);
							returnData.push({
								json: response,
								pairedItem: { item: i },
							});
						} else if (waitForCompletion) {
							// Execute and wait for completion
							const response = await teloscriptApiRequest.call(this, 'POST', `/purpose/${purposeSlug}`, body);
							returnData.push({
								json: response,
								pairedItem: { item: i },
							});
						} else {
							// Just trigger execution
							const response = await teloscriptApiRequest.call(this, 'POST', `/purpose/${purposeSlug}`, body);
							returnData.push({
								json: { 
									request_id: response.request_id, 
									endpoint_slug: purposeSlug, 
									status: 'initiated' 
								},
								pairedItem: { item: i },
							});
						}
					} else if (operation === 'listAvailable') {
						const response = await teloscriptApiRequest.call(this, 'GET', '/purpose/endpoints');
						
						if (Array.isArray(response)) {
							// Return each purpose as a separate item
							for (const purpose of response) {
								returnData.push({
									json: purpose,
									pairedItem: { item: i },
								});
							}
						} else {
							returnData.push({
								json: response,
								pairedItem: { item: i },
							});
						}
					} else if (operation === 'getDetails') {
						const purposeSlug = this.getNodeParameter('purposeSlug', i) as string;
						const response = await teloscriptApiRequest.call(this, 'GET', `/purpose/endpoints/${purposeSlug}`);
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