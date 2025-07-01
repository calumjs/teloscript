# Converting TELOSCRIPT to an n8n Package

## Overview

This guide will help you convert your TELOSCRIPT agent orchestration platform into n8n custom nodes. We'll create two main nodes:

1. **TELOSCRIPT Agent Node** - Launch agents with custom goals and MCP server configurations
2. **TELOSCRIPT Purpose Node** - Execute predefined purpose endpoints

## Project Structure

```
n8n-nodes-teloscript/
├── package.json
├── credentials/
│   └── TeloscriptApi.credentials.ts
├── nodes/
│   ├── TeloscriptAgent/
│   │   ├── TeloscriptAgent.node.ts
│   │   ├── TeloscriptAgent.node.json
│   │   └── teloscript.svg
│   └── TeloscriptPurpose/
│       ├── TeloscriptPurpose.node.ts
│       ├── TeloscriptPurpose.node.json
│       └── teloscript.svg
├── utils/
│   └── GenericFunctions.ts
└── README.md
```

## Implementation Steps

### 1. Set up the Package Structure

First, clone the n8n node starter template:

```bash
git clone https://github.com/n8n-io/n8n-nodes-starter.git n8n-nodes-teloscript
cd n8n-nodes-teloscript
```

### 2. Create the Package Configuration

Update `package.json` with your TELOSCRIPT package details:

```json
{
  "name": "n8n-nodes-teloscript",
  "version": "1.0.0",
  "description": "n8n nodes for TELOSCRIPT agent orchestration platform",
  "keywords": [
    "n8n-community-node-package"
  ],
  "license": "MIT",
  "homepage": "https://github.com/your-username/n8n-nodes-teloscript",
  "author": {
    "name": "Your Name",
    "email": "your.email@example.com"
  },
  "repository": {
    "type": "git",
    "url": "git+https://github.com/your-username/n8n-nodes-teloscript.git"
  },
  "main": "index.js",
  "scripts": {
    "build": "tsc && gulp build:icons",
    "dev": "tsc --watch",
    "format": "prettier --write .",
    "lint": "eslint .",
    "lintfix": "eslint . --fix",
    "prepublishOnly": "npm run build && npm run lint",
    "test": "jest"
  },
  "files": [
    "dist"
  ],
  "n8n": {
    "n8nNodesApiVersion": 1,
    "credentials": [
      "dist/credentials/TeloscriptApi.credentials.js"
    ],
    "nodes": [
      "dist/nodes/TeloscriptAgent/TeloscriptAgent.node.js",
      "dist/nodes/TeloscriptPurpose/TeloscriptPurpose.node.js"
    ]
  },
  "devDependencies": {
    "@types/node": "^18.16.1",
    "gulp": "^4.0.2",
    "n8n-workflow": "*",
    "typescript": "^5.0.4"
  },
  "peerDependencies": {
    "n8n-workflow": "*"
  }
}
```

### 3. Key Features to Expose

Based on your API analysis, the n8n nodes should expose:

**TELOSCRIPT Agent Node:**
- Launch agents with custom goals
- Configure MCP servers
- Set max iterations and timeout
- Stream real-time updates
- Return execution results

**TELOSCRIPT Purpose Node:**
- Execute predefined purpose endpoints
- Support for GitHub webhook handling, code analysis, research
- Custom input data
- Stream execution progress

### 4. Authentication Strategy

Your TELOSCRIPT API appears to run on localhost:8000 by default. For n8n integration, you'll need:

- Base URL configuration
- Optional API key/authentication if you add it to TELOSCRIPT
- Health check endpoint for credential validation

### 5. Node Capabilities

**Agent Node Operations:**
- Launch Agent (simple text goal)
- Launch Agent (advanced with MCP config)
- Get Agent Status
- Cancel Agent
- Stream Agent Progress

**Purpose Node Operations:**
- Execute Purpose Endpoint
- List Available Purposes
- Get Purpose Status

## Implementation Details

The implementation will include:

1. **Credentials file** for TELOSCRIPT API authentication
2. **Two node files** with full TypeScript implementation
3. **Utility functions** for API communication
4. **Icon and metadata** files
5. **Error handling and status updates**

## Installation Process

Once built, users will install your package:

```bash
# Global n8n installation
npm install -g n8n-nodes-teloscript

# Self-hosted n8n
npm install n8n-nodes-teloscript
```

## Next Steps

1. Create the credential file
2. Implement the TELOSCRIPT Agent node
3. Implement the TELOSCRIPT Purpose node
4. Add utility functions
5. Test the nodes
6. Publish to npm

This structure will allow n8n users to easily integrate with your TELOSCRIPT platform for agent orchestration and purpose execution workflows.