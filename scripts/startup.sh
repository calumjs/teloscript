#!/bin/bash

echo "🎯 TELOSCRIPT Startup - Preloading MCP Servers"

# Function to validate environment
validate_environment() {
    echo "🔍 Validating environment..."
    
    # Check if required tools are available
    local missing_tools=()
    
    command -v node >/dev/null 2>&1 || missing_tools+=("node")
    command -v npm >/dev/null 2>&1 || missing_tools+=("npm")
    command -v python >/dev/null 2>&1 || missing_tools+=("python")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo "❌ Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    echo "✅ Environment validation passed"
}

# Function to show versions
show_versions() {
    echo "📋 Environment versions:"
    echo "  - Python: $(python --version)"
    echo "  - Node.js: $(node --version)"
    echo "  - npm: $(npm --version)"
}

# Function to run MCP setup commands at startup
run_mcp_setup_commands() {
    local config_file="config/mcp_configs.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️  MCP config file not found at $config_file, skipping setup commands"
        return 0
    fi
    
    echo "🔧 Running MCP setup commands..."
    
    # Check if any configs have setup_command
    local has_setup_commands=$(jq -r '
        [.[] | select(.setup_command)] | length
    ' "$config_file" 2>/dev/null)
    
    if [[ "$has_setup_commands" == "0" ]]; then
        echo "📦 No setup commands found in MCP configs"
        return 0
    fi
    
    # Run setup command for each MCP server that has one
    jq -c '.[] | select(.setup_command) | {name: .config.name, command: .setup_command}' "$config_file" 2>/dev/null | while read -r server_data; do
        local server_name=$(echo "$server_data" | jq -r '.name')
        local command=$(echo "$server_data" | jq -r '.command')
        
        if [[ -n "$command" ]]; then
            echo "⚙️  Running setup for $server_name:"
            echo "  🔄 $command"
            if eval "$command"; then
                echo "    ✅ Setup completed successfully"
            else
                echo "    ⚠️  Warning: Setup command failed (continuing anyway)"
            fi
        fi
    done
    
    echo "✅ MCP setup commands completed"
}

# Function to verify preloaded MCP servers
verify_mcp_servers() {
    local config_file="config/mcp_configs.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️  MCP config file not found at $config_file"
        return 0
    fi
    
    echo "📁 Checking preloaded MCP packages from $config_file"
    
    # Extract all npm packages that should be installed
    # Look for configs where args contains "@modelcontextprotocol/" packages
    local packages=$(jq -r '
        .[] | 
        select(.config.command == "npx" and (.config.args | any(startswith("@modelcontextprotocol/")))) | 
        .config.args[] | select(startswith("@modelcontextprotocol/"))
    ' "$config_file" 2>/dev/null | sort -u)
    
    if [[ -z "$packages" ]]; then
        echo "📦 No @modelcontextprotocol packages in config"
        return 0
    fi
    
    echo "🔍 Verifying preloaded packages:"
    local missing_packages=()
    local installed_packages=()
    
    while IFS= read -r package; do
        if [[ -n "$package" ]]; then
            if npm list -g --depth=0 "$package" >/dev/null 2>&1; then
                installed_packages+=("$package")
                echo "  ✅ $package"
            else
                missing_packages+=("$package")
                echo "  ❌ $package (not found)"
            fi
        fi
    done <<< "$packages"
    
    if [[ ${#installed_packages[@]} -gt 0 ]]; then
        echo "✅ ${#installed_packages[@]} packages ready for fast startup"
    fi
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        echo "⚠️  ${#missing_packages[@]} packages not preloaded (will use npx -y fallback):"
        printf '  - %s\n' "${missing_packages[@]}"
    fi
}

# Main startup sequence
main() {
    echo "════════════════════════════════════════════════════════════════"
    echo "🎯 TELOSCRIPT - Purposeful Agent Orchestration System"
    echo "════════════════════════════════════════════════════════════════"
    
    validate_environment
    show_versions
    
    echo ""
    verify_mcp_servers
    
    echo ""
    run_mcp_setup_commands
    
    echo ""
    echo "🚀 Starting TELOSCRIPT application..."
    echo "════════════════════════════════════════════════════════════════"
    
    # Start the main application
    exec python main.py
}

# Handle signals gracefully
trap 'echo "⏹️  Received shutdown signal, stopping TELOSCRIPT..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@" 