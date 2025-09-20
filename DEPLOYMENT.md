# Tamil News Summarizer - Deployment Guide

## Quick Start (Local Development)

### Windows
1. Run `start.bat` - This will start both backend and frontend automatically
2. Open http://localhost:3000 in your browser

### Manual Setup
```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Production Deployment Options

### 1. Heroku Deployment

#### Backend (FastAPI)
```bash
# Create Procfile in backend/
echo "web: uvicorn main:app --host 0.0.0.0 --port $PORT" > backend/Procfile

# Deploy
cd backend
git init
heroku create your-app-backend
git add .
git commit -m "Deploy backend"
git push heroku main
```

#### Frontend (Next.js)
```bash
# Deploy to Vercel (recommended)
cd frontend
npm install -g vercel
vercel --prod

# Or Netlify
npm run build
# Upload dist folder to Netlify
```

### 2. Railway Deployment

#### Backend
1. Connect GitHub repo to Railway
2. Select backend folder
3. Add environment variables if needed
4. Deploy automatically

#### Frontend
1. Deploy to Vercel/Netlify
2. Update API URL to Railway backend URL

### 3. DigitalOcean App Platform

```yaml
# app.yaml
name: tamil-news-summarizer
services:
- name: backend
  source_dir: /backend
  github:
    repo: your-username/tamil-news-summarizer
    branch: main
  run_command: uvicorn main:app --host 0.0.0.0 --port $PORT
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  
- name: frontend
  source_dir: /frontend
  github:
    repo: your-username/tamil-news-summarizer
    branch: main
  run_command: npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
```

### 4. AWS Deployment

#### Using AWS Elastic Beanstalk
```bash
# Backend
cd backend
pip install awsebcli
eb init
eb create tamil-news-backend
eb deploy

# Frontend - Deploy to S3 + CloudFront
cd frontend
npm run build
aws s3 sync out/ s3://your-bucket-name
```

### 5. Google Cloud Platform

```bash
# Backend - Cloud Run
cd backend
gcloud run deploy tamil-news-backend --source .

# Frontend - Firebase Hosting
cd frontend
npm install -g firebase-tools
firebase init hosting
npm run build
firebase deploy
```

### 6. Docker Deployment

#### Create Dockerfiles

**Backend Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Environment Variables

### Backend (.env)
```
# Optional - for production optimizations
MODEL_CACHE_DIR=/app/cache
MAX_WORKERS=4
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
# For production, change to your backend URL
```

## Performance Optimization

### Backend
- Use gunicorn for production: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker`
- Enable model caching
- Use Redis for session storage (optional)

### Frontend
- Enable Next.js static optimization
- Use CDN for assets
- Enable compression

## Monitoring & Logging

### Add to backend/main.py:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to endpoints
logger.info(f"Summarization request: {len(news.text)} chars")
```

## Security Considerations

1. **CORS**: Update CORS origins for production
2. **Rate Limiting**: Add rate limiting middleware
3. **API Keys**: Secure MyMemory API usage
4. **HTTPS**: Always use HTTPS in production

## Troubleshooting

### Common Issues:
1. **Model Loading**: Ensure sufficient RAM (4GB+)
2. **CORS Errors**: Update backend CORS settings
3. **Port Conflicts**: Change ports if 3000/8000 are occupied
4. **Dependencies**: Use exact versions in requirements.txt

### Memory Requirements:
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- Model size: ~2GB

## Cost Estimation

### Free Tier Options:
- **Heroku**: 550 hours/month free
- **Railway**: $5/month after free tier
- **Vercel**: Unlimited for personal projects
- **Netlify**: 100GB bandwidth free

### Paid Options:
- **DigitalOcean**: $5-10/month
- **AWS**: $10-20/month (depending on usage)
- **GCP**: Similar to AWS

## Backup & Recovery

1. **Code**: Use Git with GitHub/GitLab
2. **Models**: Models auto-download from Hugging Face
3. **Data**: No persistent data storage needed

## Scaling

### Horizontal Scaling:
- Use load balancer
- Multiple backend instances
- CDN for frontend

### Vertical Scaling:
- Increase RAM for model performance
- Use GPU instances for faster inference