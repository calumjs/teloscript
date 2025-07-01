# Docker Deployment Guide for n8n TELOSCRIPT Package

## Overview

Your TELOSCRIPT n8n package can be deployed using Docker in several ways. This guide covers all the deployment options, from simple installations to production-ready setups.

## Deployment Methods

### 🎯 **Method 1: Community Package Installation (Recommended)**

Once your package is published to npm, users can install it in their Docker n8n instances:

#### Option A: Environment Variable Method
```yaml
# docker-compose.yml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_COMMUNITY_PACKAGES_ENABLED=true
      - N8N_NODES_INCLUDE=n8n-nodes-teloscript
    volumes:
      - n8n_data:/home/node/.n8n
    restart: unless-stopped

volumes:
  n8n_data:
```

#### Option B: Custom Dockerfile Method
```dockerfile
FROM n8nio/n8n:latest
USER root
RUN npm install -g n8n-nodes-teloscript
USER node
```

### 🛠️ **Method 2: Private/Development Installation**

For private development or internal use without publishing to npm:

#### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'
services:
  n8n:
    build: .
    ports:
      - "5678:5678"
    environment:
      - TELOSCRIPT_BASE_URL=http://teloscript:8000  # If running TELOSCRIPT in Docker
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n-nodes-teloscript/dist:/home/node/.n8n/custom/node_modules/n8n-nodes-teloscript
    depends_on:
      - teloscript
    restart: unless-stopped

  teloscript:
    # Your TELOSCRIPT service configuration
    build: ../teloscript  # Path to your TELOSCRIPT project
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped

volumes:
  n8n_data:
```

#### Dockerfile for Custom Installation
```dockerfile
FROM n8nio/n8n:latest

# Switch to root to install packages
USER root

# Install your custom node (if you have the built dist folder)
COPY ./dist /home/node/.n8n/custom/node_modules/n8n-nodes-teloscript

# Or install from npm if published
# RUN npm install -g n8n-nodes-teloscript

# Switch back to node user
USER node

# Expose n8n port
EXPOSE 5678
```

### 🏭 **Method 3: Production-Ready Setup**

#### Complete Stack with TELOSCRIPT + n8n + Database

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: n8n
      POSTGRES_USER: n8n
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  n8n:
    build: .
    ports:
      - "5678:5678"
    environment:
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${DB_PASSWORD}
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - WEBHOOK_URL=https://your-domain.com/
      - GENERIC_TIMEZONE=America/New_York
      - TELOSCRIPT_BASE_URL=http://teloscript:8000
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - postgres
      - teloscript
    restart: unless-stopped

  teloscript:
    build: ../teloscript
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://teloscript:${TELOSCRIPT_DB_PASSWORD}@postgres:5432/teloscript
    volumes:
      - teloscript_data:/data
    depends_on:
      - postgres
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - n8n
    restart: unless-stopped

volumes:
  postgres_data:
  n8n_data:
  teloscript_data:
```

### 🚀 **Method 4: Kubernetes Deployment**

For enterprise-scale deployments:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: n8n-teloscript
spec:
  replicas: 2
  selector:
    matchLabels:
      app: n8n-teloscript
  template:
    metadata:
      labels:
        app: n8n-teloscript
    spec:
      containers:
      - name: n8n
        image: your-registry/n8n-with-teloscript:latest
        ports:
        - containerPort: 5678
        env:
        - name: TELOSCRIPT_BASE_URL
          value: "http://teloscript-service:8000"
        - name: DB_TYPE
          value: "postgresdb"
        - name: DB_POSTGRESDB_HOST
          value: "postgres-service"
        volumeMounts:
        - name: n8n-data
          mountPath: /home/node/.n8n
      volumes:
      - name: n8n-data
        persistentVolumeClaim:
          claimName: n8n-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: n8n-service
spec:
  selector:
    app: n8n-teloscript
  ports:
  - port: 80
    targetPort: 5678
  type: LoadBalancer
```

## 🔧 **Environment Configuration**

### Required Environment Variables

```bash
# .env file
# n8n Configuration
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your-secure-password
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=https
WEBHOOK_URL=https://your-domain.com/

# Database (for production)
DB_TYPE=postgresdb
DB_POSTGRESDB_HOST=postgres
DB_POSTGRESDB_PORT=5432
DB_POSTGRESDB_DATABASE=n8n
DB_POSTGRESDB_USER=n8n
DB_POSTGRESDB_PASSWORD=your-db-password

# TELOSCRIPT Configuration
TELOSCRIPT_BASE_URL=http://teloscript:8000
OPENAI_API_KEY=your-openai-api-key

# Security
GENERIC_TIMEZONE=America/New_York
N8N_SECURE_COOKIE=true
```

## 📋 **Step-by-Step Deployment**

### For Community Package Users:

1. **Create docker-compose.yml:**
```bash
curl -O https://raw.githubusercontent.com/your-username/n8n-nodes-teloscript/main/docker-compose.yml
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. **Deploy:**
```bash
docker-compose up -d
```

4. **Install TELOSCRIPT package in n8n UI:**
   - Go to Settings → Community Nodes
   - Install `n8n-nodes-teloscript`

### For Private/Development:

1. **Clone and build:**
```bash
git clone https://github.com/your-username/n8n-nodes-teloscript.git
cd n8n-nodes-teloscript
npm run build
```

2. **Deploy with volume mapping:**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

## 🔍 **Networking Considerations**

### Service Communication
```yaml
# If TELOSCRIPT and n8n are in the same Docker network:
services:
  n8n:
    environment:
      - TELOSCRIPT_BASE_URL=http://teloscript:8000  # Use service name
  
  teloscript:
    # Your TELOSCRIPT configuration
```

### External TELOSCRIPT Instance
```yaml
# If TELOSCRIPT is running separately:
services:
  n8n:
    environment:
      - TELOSCRIPT_BASE_URL=https://your-teloscript-instance.com
    extra_hosts:
      - "teloscript.local:host-gateway"  # For local development
```

## 🛡️ **Security & Production Considerations**

### SSL/TLS Configuration
```yaml
# nginx.conf for reverse proxy
events {
    worker_connections 1024;
}

http {
    upstream n8n {
        server n8n:5678;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        location / {
            proxy_pass http://n8n;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Health Checks
```yaml
# Add to your services:
healthcheck:
  test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:5678/healthz"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 📊 **Monitoring & Logging**

### Docker Compose with Monitoring
```yaml
services:
  # ... n8n and teloscript services ...
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔄 **Updates & Maintenance**

### Updating the Package
```bash
# For community package
docker-compose exec n8n npm update -g n8n-nodes-teloscript
docker-compose restart n8n

# For private package with volume mapping
cd n8n-nodes-teloscript
git pull
npm run build
docker-compose restart n8n
```

### Backup Strategy
```bash
# Backup n8n data
docker-compose exec postgres pg_dump -U n8n n8n > backup.sql

# Backup volumes
docker run --rm -v n8n_data:/data -v $(pwd):/backup alpine tar czf /backup/n8n-backup.tar.gz /data
```

## 🐛 **Troubleshooting**

### Common Issues

1. **Node not appearing:**
```bash
# Check if package is installed
docker-compose exec n8n npm list -g n8n-nodes-teloscript

# Check n8n logs
docker-compose logs n8n
```

2. **Connection to TELOSCRIPT fails:**
```bash
# Test network connectivity
docker-compose exec n8n wget -qO- http://teloscript:8000/health

# Check TELOSCRIPT logs
docker-compose logs teloscript
```

3. **Permission issues:**
```bash
# Fix volume permissions
docker-compose exec n8n chown -R node:node /home/node/.n8n
```

This Docker setup gives you production-ready deployment options for your TELOSCRIPT n8n package, from simple single-container setups to full enterprise deployments with monitoring and high availability!