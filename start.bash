#!/bin/bash

# Start Admin services
echo "Starting Admin FastAPI service..."
python AdminFastAPI.py &

# Wait for Admin FastAPI to start (adjust sleep time if needed)
sleep 3

echo "Starting Admin Web UI..."
python AdminWebui.py &

# Wait for Admin Web UI to start
sleep 3

# Start Visitor services
echo "Starting Visitor FastAPI service..."
python VisitorFastAPI.py &

# Wait for Visitor FastAPI to start
sleep 3

echo "Starting Visitor Web UI..."
python VisitorWebui.py &

echo "All services started in the background."
echo ""
echo "Note: Press Ctrl+C to stop all services."

# Keep the script running to keep background processes alive
wait