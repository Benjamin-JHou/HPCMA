# HPCMA Deployment Configuration

## Container & Environment Setup

This directory contains deployment configurations for the Hypertension Pan-Comorbidity Multi-Modal Atlas.

---

## 📁 Contents

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker container configuration (CPU-only) |
| `environment.yml` | Conda environment specification |
| `README.md` | This documentation |

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t hpcma:latest .
```

### Run Container

```bash
# Basic run
docker run -p 8000:8000 hpcma:latest

# With volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/atlas_resource:/app/atlas_resource \
  -v $(pwd)/models:/app/models \
  hpcma:latest

# Production mode
docker run -d \
  --name hpcma-api \
  -p 8000:8000 \
  --restart unless-stopped \
  hpcma:latest
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  hpcma:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./atlas_resource:/app/atlas_resource:ro
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models
      - CONFIG_PATH=/app/config
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 🐍 Conda Environment

### Create Environment

```bash
conda env create -f environment.yml
conda activate mmrp-clinical-ai
```

### Update Environment

```bash
conda env update -f environment.yml --prune
```

### Export Current Environment

```bash
conda env export > environment.yml
```

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel/AMD x86_64)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **OS**: Linux/macOS/Windows (WSL2)

### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 16 GB+
- **Storage**: 20 GB SSD
- **Network**: Public IP for API access

### Software Dependencies
- Docker 20.10+ (for containerization)
- Conda 4.10+ (for environment management)
- Python 3.9+

---

## 🚀 Deployment Scenarios

### Scenario 1: Local Development

```bash
# Using Conda
conda activate mmrp-clinical-ai
python -m src.inference.api_server

# Using Docker
docker run -p 8000:8000 -v $(pwd):/app hpcma:latest
```

### Scenario 2: Server Deployment

```bash
# Production deployment
docker run -d \
  --name hpcma-production \
  -p 80:8000 \
  --restart always \
  -e LOG_LEVEL=WARNING \
  hpcma:latest
```

### Scenario 3: Cloud Deployment

**AWS ECS**:
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker tag hpcma:latest $ECR_URL/hpcma:latest
docker push $ECR_URL/hpcma:latest
```

**Google Cloud Run**:
```bash
# Deploy to Cloud Run
gcloud run deploy hpcma \
  --image hpcma:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🔒 Security Considerations

### Docker Security
- Container runs as non-root user
- Minimal base image (python:3.9-slim)
- No sensitive data in image layers
- Health checks enabled

### Environment Variables
```bash
# Required
MODEL_PATH=/app/models
CONFIG_PATH=/app/config

# Optional
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
PORT=8000
HOST=0.0.0.0
```

### Network Security
- Bind to localhost only for local dev: `-p 127.0.0.1:8000:8000`
- Use reverse proxy (nginx/traefik) for production
- Enable HTTPS/TLS in production

---

## 📊 Resource Monitoring

### Docker Stats
```bash
docker stats hpcma
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
```bash
# View logs
docker logs hpcma

# Follow logs
docker logs -f hpcma
```

---

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
- name: Build Docker Image
  run: docker build -t hpcma:${{ github.sha }} .

- name: Test Container
  run: |
    docker run -d -p 8000:8000 hpcma:${{ github.sha }}
    sleep 10
    curl -f http://localhost:8000/health || exit 1
```

---

## 🆘 Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs hpcma

# Check resource limits
docker system df
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Use different port
docker run -p 8080:8000 hpcma:latest
```

### Memory Issues
```bash
# Increase Docker memory limit
docker run -m 4g hpcma:latest
```

---

## 📚 Additional Resources

- **Main README**: `../README.md`
- **API Documentation**: Access `/docs` endpoint when running
- **Validation Guide**: `../VALIDATION_WHITEPAPER.md`

---

**Last Updated**: February 2025 | **Version**: 1.0.0
