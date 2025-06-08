#!/bin/bash

echo "🔧 Preloading MCP Servers during Docker build"

# Function to run setup commands for MCP servers
run_setup_commands() {
    local config_file="config/mcp_configs.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️  MCP config file not found at $config_file, skipping setup commands"
        return 0
    fi
    
    echo "🔍 Checking for setup commands in MCP configurations..."
    
    # Check if any configs have setup_command
    local has_setup_commands=$(jq -r '
        [.[] | select(.setup_command)] | length
    ' "$config_file" 2>/dev/null)
    
    if [[ "$has_setup_commands" == "0" ]]; then
        echo "📦 No setup commands found in MCP configs"
        return 0
    fi
    
    echo "🔧 Running setup commands for MCP servers..."
    
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
                echo "    ⚠️  Warning: Setup command failed"
            fi
        fi
    done
}

# Function to install system packages for MCP servers
install_system_packages() {
    local config_file="config/mcp_configs.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️  MCP config file not found at $config_file, skipping system packages"
        return 0
    fi
    
    echo "🔍 Checking for system packages in MCP configurations..."
    
    # Collect all unique system packages from all configs
    local packages=$(jq -r '
        [.[] | select(.system_packages) | .system_packages[]] | 
        sort | unique | .[]
    ' "$config_file" 2>/dev/null)
    
    if [[ -z "$packages" ]]; then
        echo "📦 No system packages found in MCP configs"
        return 0
    fi
    
    echo "📦 Found system packages to install:"
    echo "$packages" | sed 's/^/  - /'
    
    echo "⏳ Installing system packages with apt-get..."
    local package_list=""
    while IFS= read -r package; do
        if [[ -n "$package" ]]; then
            package_list="$package_list $package"
        fi
    done <<< "$packages"
    
    if [[ -n "$package_list" ]]; then
        echo "🔄 Running: apt-get update && apt-get install -y$package_list"
        if apt-get update && apt-get install -y $package_list; then
            echo "✅ Successfully installed system packages"
            # Clean up apt cache
            apt-get clean && rm -rf /var/lib/apt/lists/*
        else
            echo "⚠️  Warning: Some system packages failed to install"
            return 1
        fi
    fi
}

# Function to extract and install npm packages from MCP config
preload_mcp_servers() {
    local config_file="config/mcp_configs.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "⚠️  MCP config file not found at $config_file, skipping preload"
        return 0
    fi
    
    echo "📁 Reading MCP configurations from $config_file"
    
    # Extract all npm packages that need to be installed
    # Look for configs where command is "npx" and args contains "@modelcontextprotocol/" packages
    local packages=$(jq -r '
        .[] | 
        select(.config.command == "npx" and (.config.args | any(startswith("@modelcontextprotocol/")))) | 
        .config.args[] | select(startswith("@modelcontextprotocol/"))
    ' "$config_file" 2>/dev/null | sort -u)
    
    if [[ -z "$packages" ]]; then
        echo "📦 No @modelcontextprotocol packages found in MCP config, skipping preload"
        return 0
    fi
    
    echo "📦 Found packages to preload:"
    echo "$packages" | sed 's/^/  - /'
    
    echo "⏳ Installing packages globally..."
    local install_list=""
    while IFS= read -r package; do
        if [[ -n "$package" ]]; then
            install_list="$install_list $package"
        fi
    done <<< "$packages"
    
    if [[ -n "$install_list" ]]; then
        echo "🔄 Running: npm install -g$install_list"
        if npm install -g $install_list; then
            echo "✅ Successfully preloaded MCP servers"
            npm cache clean --force >/dev/null 2>&1
            
            # List what was installed
            echo "📋 Installed packages:"
            npm list -g --depth=0 2>/dev/null | grep "@modelcontextprotocol" | sed 's/^/  /'
        else
            echo "⚠️  Warning: Some packages failed to install during preload"
            return 1
        fi
    fi
}

# Main preload function
main() {
    local mode=${1:-"full"}
    
    if [[ "$mode" == "setup-only" ]]; then
        echo "════════════════════════════════════════════════════════════════"
        echo "🔧 MCP Setup Commands (Agent User)"
        echo "════════════════════════════════════════════════════════════════"
        
        # Only run setup commands
        run_setup_commands
        
        echo ""
        echo "✅ MCP setup commands completed"
        echo "════════════════════════════════════════════════════════════════"
        return 0
    fi
    
    echo "════════════════════════════════════════════════════════════════"
    echo "🔧 MCP Server Preloading (Docker Build Phase)"
    echo "════════════════════════════════════════════════════════════════"
    
    # Check if required tools are available
    if ! command -v node >/dev/null 2>&1; then
        echo "❌ Node.js not found"
        exit 1
    fi
    
    if ! command -v npm >/dev/null 2>&1; then
        echo "❌ npm not found"
        exit 1
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        echo "❌ jq not found"
        exit 1
    fi
    
    echo "📋 Build environment:"
    echo "  - Node.js: $(node --version)"
    echo "  - npm: $(npm --version)"
    
    echo ""
    # Install system packages first
    install_system_packages
    
    echo ""
    # Install MCP packages
    preload_mcp_servers
    
    echo ""
    echo "✅ MCP preloading completed"
    echo "════════════════════════════════════════════════════════════════"
}

# Run main function
main "$@" 