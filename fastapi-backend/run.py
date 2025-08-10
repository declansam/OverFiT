#!/usr/bin/env python3
"""
Production-ready runner for the FastAPI backend
"""

import uvicorn
from config import HOST, PORT, DEBUG

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug",
        access_log=True
    )
