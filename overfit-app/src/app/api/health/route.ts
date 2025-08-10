import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  try {
    // Check FastAPI backend health
    const response = await fetch(`${FASTAPI_BASE_URL}/health`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        {
          status: "unhealthy",
          message: "FastAPI backend is not responding",
          fastapi_url: FASTAPI_BASE_URL,
        },
        { status: 503 }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      status: "healthy",
      message: "Next.js API and FastAPI backend are both running",
      fastapi_status: data.status,
      fastapi_message: data.message,
      fastapi_url: FASTAPI_BASE_URL,
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "unhealthy",
        message: "Failed to connect to FastAPI backend",
        error: error instanceof Error ? error.message : "Unknown error",
        fastapi_url: FASTAPI_BASE_URL,
      },
      { status: 503 }
    );
  }
}
