# Development Setup Guide

## Prerequisites

### System Requirements
- **Operating System**: [Windows 10+/macOS 12+/Ubuntu 20.04+]
- **Memory**: Minimum 8GB RAM, Recommended 16GB+
- **Storage**: Minimum 20GB free space
- **Display**: 1920x1080 minimum resolution

### Required Software
1. **Git**: Version control system
   - Download: https://git-scm.com/
   - Version: 2.30.0 or higher

2. **Node.js**: JavaScript runtime
   - Download: https://nodejs.org/
   - Version: 16.14.0 or higher
   - Includes npm package manager

3. **Python**: Programming language
   - Download: https://python.org/
   - Version: 3.9.0 or higher
   - Includes pip package manager

4. **Docker**: Containerization platform
   - Download: https://docker.com/
   - Version: 20.10.0 or higher
   - Includes Docker Compose

5. **Code Editor**: Development environment
   - **Visual Studio Code** (Recommended)
     - Download: https://code.visualstudio.com/
     - Required Extensions: [List extensions]

   - **Alternative**: IntelliJ IDEA, PyCharm, etc.

### Optional Software
- **PostgreSQL**: Database server (if running locally)
- **Redis**: Caching server (if running locally)
- **Postman**: API testing tool
- **DBeaver**: Database client

## Installation Steps

### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/your-org/your-project.git

# Navigate to project directory
cd your-project

# Check out the main development branch
git checkout develop
```

### 2. Install Dependencies

#### Frontend Dependencies
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list --depth=0
```

#### Backend Dependencies
```bash
# Navigate to backend directory
cd ../backend

# Install Python dependencies
pip install -r requirements.txt

# For development with hot reload
pip install -r requirements-dev.txt

# Verify installation
pip list
```

#### Database Setup (Local)
```bash
# Start PostgreSQL service
# On macOS with Homebrew
brew services start postgresql

# On Ubuntu
sudo service postgresql start

# Create database
createdb your_project_db

# Run migrations
cd backend
python manage.py migrate
```

### 3. Environment Configuration

#### Copy Environment Files
```bash
# Frontend
cp frontend/.env.example frontend/.env

# Backend
cp backend/.env.example backend/.env

# Docker
cp docker-compose.override.yml.example docker-compose.override.yml
```

#### Configure Environment Variables

**Frontend (.env)**:
```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_API_TIMEOUT=5000

# Authentication
REACT_APP_AUTH_DOMAIN=your-auth-domain.auth0.com
REACT_APP_AUTH_CLIENT_ID=your-client-id
REACT_APP_AUTH_AUDIENCE=your-api-identifier

# Feature Flags
REACT_APP_ENABLE_DEBUG=true
REACT_APP_ENABLE_ANALYTICS=false
```

**Backend (.env)**:
```env
# Django Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/your_project_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Email Configuration
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# API Keys
STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key
STRIPE_SECRET_KEY=sk_test_your_secret_key

# Third-party Services
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

### 4. Database Migration

#### Run Migrations
```bash
# Backend directory
cd backend

# Run Django migrations
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser
```

#### Seed Database (Optional)
```bash
# Run data seeding script
python manage.py seed_data

# Or load fixtures
python manage.py loaddata fixtures/sample_data.json
```

### 5. Start Development Servers

#### Option A: Using Docker (Recommended)
```bash
# Root directory
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Option B: Manual Startup

**Start Backend**:
```bash
# Backend directory
cd backend

# Start Django development server
python manage.py runserver

# Server will start on http://localhost:8000
```

**Start Frontend**:
```bash
# Frontend directory
cd frontend

# Start React development server
npm start

# Server will start on http://localhost:3000
```

**Start Additional Services**:
```bash
# Start Redis (if not using Docker)
redis-server

# Start Celery worker (for background tasks)
cd backend
celery -A your_project worker --loglevel=info

# Start Celery beat (for scheduled tasks)
celery -A your_project beat --loglevel=info
```

### 6. Verify Installation

#### Health Checks
```bash
# API Health Check
curl http://localhost:8000/api/v1/health/

# Database Connection Test
python manage.py shell
>>> from django.db import connection
>>> connection.ensure_connection()
>>> print("Database connected successfully")

# Redis Connection Test
python manage.py shell
>>> from redis import Redis
>>> r = Redis(host='localhost', port=6379, db=0)
>>> r.ping()
True
```

#### Run Tests
```bash
# Backend tests
cd backend
python manage.py test

# Frontend tests
cd frontend
npm test

# End-to-end tests
cd e2e
npm run test:e2e
```

## Development Workflow

### Daily Development
1. **Pull latest changes**: `git pull origin develop`
2. **Update dependencies**: `npm install && pip install -r requirements.txt`
3. **Run migrations**: `python manage.py migrate`
4. **Start servers**: `docker-compose up -d` or manual startup
5. **Run tests**: `python manage.py test && npm test`

### Code Style and Linting
```bash
# Backend linting
cd backend
flake8 .
black --check .

# Frontend linting
cd frontend
npm run lint

# Fix formatting automatically
npm run lint:fix
```

### Database Management
```bash
# Create migration
python manage.py makemigrations

# Show migration SQL
python manage.py sqlmigrate app_name migration_number

# Revert migration
python manage.py migrate app_name previous_migration
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using a port
lsof -i :8000

# Kill process using port
kill -9 PID
```

#### Database Connection Issues
```bash
# Reset PostgreSQL
brew services restart postgresql

# Recreate database
dropdb your_project_db
createdb your_project_db
python manage.py migrate
```

#### Node Modules Issues
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear npm cache
npm cache clean --force
```

#### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose up --build

# View container logs
docker-compose logs service_name

# Access container shell
docker-compose exec service_name bash
```

### Getting Help
- **Documentation**: [Link to internal docs]
- **Team Chat**: [Slack/Discord channel]
- **Issue Tracker**: [GitHub Issues/Jira]
- **Code Review**: [Pull Request guidelines]

## Next Steps
1. Read the [Development Guide](development_guide.md)
2. Review [Coding Standards](coding_standards.md)
3. Set up your IDE with recommended extensions
4. Join the development team chat
5. Review open issues and pick your first task