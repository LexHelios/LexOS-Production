#!/bin/bash

# LexOS Genesis V3 - Production vLLM Deployment Script
# For H100 GPU deployment with production-grade configuration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime is not available or GPU is not accessible"
        exit 1
    fi
    
    log_info "All prerequisites met"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    directories=(
        "./models"
        "./logs"
        "./config"
        "./static"
        "./uploads"
        "./lexos_memory"
        "./ssl"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    log_info "Directories created"
}

# Generate SSL certificates for production
generate_ssl_certificates() {
    log_info "Generating self-signed SSL certificates..."
    
    if [[ ! -f ./ssl/cert.pem || ! -f ./ssl/key.pem ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout ./ssl/key.pem -out ./ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=LexOS/CN=lexos.local"
        chmod 600 ./ssl/key.pem
        log_info "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Create production configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Create .env.production if it doesn't exist
    if [[ ! -f .env.production ]]; then
        cat > .env.production << 'EOF'
# Production Environment Variables
POSTGRES_PASSWORD=lexos-postgres-$(openssl rand -hex 16)
REDIS_PASSWORD=lexos-redis-$(openssl rand -hex 16)
CHROMA_TOKEN=lexos-chroma-token-$(openssl rand -hex 16)
VLLM_API_KEY=token-lexos-vllm-$(openssl rand -hex 16)
JWT_SECRET=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
EOF
        chmod 600 .env.production
        log_info "Created .env.production with secure passwords"
    fi
    
    # Create nginx.conf
    cat > nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100m;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=general:10m rate=50r/s;
    
    upstream lexos_backend {
        least_conn;
        server lexos-core:8080 max_fails=3 fail_timeout=30s;
    }
    
    upstream vllm_backend {
        least_conn;
        server vllm:8000 max_fails=3 fail_timeout=30s;
    }
    
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name _;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        
        location / {
            limit_req zone=general burst=20 nodelay;
            proxy_pass http://lexos_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        location /api/ {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://lexos_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /vllm/ {
            limit_req zone=api burst=5 nodelay;
            rewrite ^/vllm/(.*) /$1 break;
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /ws {
            proxy_pass http://lexos_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    # Create prometheus.yml
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lexos-core'
    static_configs:
      - targets: ['lexos-core:8080']
    
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    
    log_info "Configuration files created"
}

# Stop existing services
stop_existing_services() {
    log_info "Stopping existing services..."
    
    # Stop all containers
    docker compose down -v 2>/dev/null || true
    docker compose -f docker-compose.vllm.yml down -v 2>/dev/null || true
    
    # Clean up any dangling containers
    docker container prune -f
    
    log_info "Existing services stopped"
}

# Build production image
build_production_image() {
    log_info "Building production Docker image..."
    
    docker build -f Dockerfile.production -t lexos-core:production .
    
    log_info "Production image built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying production services..."
    
    # Load environment variables properly
    if [[ -f .env.production ]]; then
        set -o allexport
        source .env.production
        set +o allexport
    fi
    
    # Start services with production configuration
    docker compose -f docker-compose.vllm.yml up -d
    
    log_info "Services deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    services=(
        "vllm:8001"
        "redis:6379"
        "postgres:5432"
        "chromadb:8000"
        "lexos-core:8080"
    )
    
    for service in "${services[@]}"; do
        service_name="${service%%:*}"
        service_port="${service##*:}"
        
        log_info "Waiting for $service_name on port $service_port..."
        
        timeout=300
        while ! docker exec lexos-$service_name curl -s -f http://localhost:$service_port/health 2>/dev/null && [ $timeout -gt 0 ]; do
            sleep 5
            ((timeout-=5))
        done
        
        if [ $timeout -le 0 ]; then
            log_error "$service_name failed to become healthy"
            exit 1
        fi
        
        log_info "$service_name is healthy"
    done
}

# Download vLLM model
download_vllm_model() {
    log_info "Downloading vLLM model (this may take a while)..."
    
    docker exec lexos-vllm python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/models')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/models')
print('Model downloaded successfully')
" || log_warn "Model download might be handled by vLLM automatically"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    docker compose -f docker-compose.vllm.yml ps
    echo
    log_info "Services are available at:"
    echo "  - LexOS API: https://localhost/"
    echo "  - vLLM API: https://localhost/vllm/"
    echo "  - Prometheus: http://localhost:9090"
    echo
    log_info "Default credentials are in .env.production"
}

# Main deployment flow
main() {
    echo "========================================="
    echo "LexOS Genesis V3 - Production Deployment"
    echo "========================================="
    echo
    
    check_prerequisites
    create_directories
    generate_ssl_certificates
    create_config_files
    stop_existing_services
    build_production_image
    deploy_services
    wait_for_services
    download_vllm_model
    show_status
    
    log_info "Production deployment completed successfully!"
}

# Run main function
main "$@"