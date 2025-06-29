# Purpose Endpoint Learning Feature

## Overview

The purpose endpoint learning feature automatically captures and stores insights from each execution to improve future performance. This feature uses the system's own LLM capabilities to analyze execution patterns and build up institutional knowledge.

## How It Works

### 1. Learning Capture
- After each purpose endpoint execution (both successful and failed), the system automatically analyzes the execution
- It creates a learning analysis prompt that includes:
  - Execution details (status, timing, iterations used)
  - Input data patterns
  - Results and any errors
  - Current accumulated learnings

### 2. Learning Analysis
- A lightweight agent analyzes the execution using a dedicated prompt
- The analysis focuses on:
  - What worked well (for successful executions)
  - What didn't work (for failures or slow executions)
  - Input data patterns
  - Performance insights
  - Suggested improvements

### 3. Learning Storage
- New learnings are combined with existing learnings
- The updated learnings are stored in the purpose endpoint configuration
- The configuration is automatically saved to the JSON file for persistence

### 4. Learning Application
- When building prompts for future executions, existing learnings are included
- The prompt includes: "Use these learnings to improve your approach and avoid previous mistakes"
- This helps the LLM make better decisions based on historical performance

## Features Added

### Data Model Changes
- Added `learnings` field to `PurposeEndpointConfig` as a simple text field
- Field stores accumulated insights in natural language format

### Core Functionality
- `_capture_learnings()`: Analyzes execution and generates new learnings
- `_build_learning_analysis_prompt()`: Creates analysis prompt for the LLM
- `_combine_learnings()`: Merges new learnings with existing ones
- Updated `_build_prompt()`: Includes learnings in execution prompts
- Updated `execute_endpoint()`: Calls learning capture after each execution

### API Endpoints
- `GET /purpose/endpoints/{slug}/learnings`: View learnings for an endpoint
- `DELETE /purpose/endpoints/{slug}/learnings`: Clear learnings for an endpoint

## Benefits

1. **Continuous Improvement**: Each execution makes the system smarter
2. **Pattern Recognition**: Identifies common input patterns and optimal approaches
3. **Error Avoidance**: Learns from failures to prevent repeat mistakes
4. **Performance Optimization**: Discovers timing and configuration insights
5. **No Manual Intervention**: Fully automatic learning process

## Example Learning Process

1. **First Execution**: No prior learnings, execution proceeds normally
2. **Learning Analysis**: System analyzes what worked/didn't work
3. **Learning Storage**: Insights stored in endpoint configuration
4. **Second Execution**: Prompt includes previous learnings
5. **Improved Performance**: LLM uses historical insights to perform better
6. **Continuous Cycle**: Each execution adds to the knowledge base

## Technical Implementation

- **Simple Text Storage**: Learnings stored as plain text for maximum flexibility
- **LLM-Driven Analysis**: Uses the system's own capabilities for learning analysis
- **Automatic Persistence**: Changes saved to configuration file immediately
- **Lightweight Design**: Minimal performance impact on main execution
- **Error Handling**: Learning failures don't affect main endpoint execution

## Usage

The learning feature works automatically with no configuration required. Users can:

- View learnings via API: `GET /purpose/endpoints/my-endpoint/learnings`
- Clear learnings if needed: `DELETE /purpose/endpoints/my-endpoint/learnings`
- Monitor learning accumulation through the updated_at timestamp

The feature is designed to be transparent and require no maintenance while continuously improving system performance.