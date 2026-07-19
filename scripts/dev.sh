#!/bin/bash

# ============================================================================
# QUANT X LOCAL DEVELOPMENT SCRIPT
# Run this to start both backend and frontend
# ============================================================================

set -e

echo "🚀 Starting Quant X Development Environment"
echo "============================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "   Copy .env.example to .env and fill in your values"
    echo "   Run: cp .env.example .env"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo ""
echo "📦 Starting Backend (FastAPI)..."
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Backend URL: http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"

# Wait for backend to start
sleep 3

# Start Frontend
echo ""
echo "🎨 Starting Frontend (Next.js)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
echo "   Frontend URL: http://localhost:3000"

echo ""
echo "============================================"
echo "✅ Quant X is running!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "============================================"

# Wait for processes
wait
