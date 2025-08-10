import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  try {
    // Forward request to FastAPI backend
    const response = await fetch(`${FASTAPI_BASE_URL}/api/get-molecules`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || "Failed to retrieve molecules" },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      molecules: data.molecules,
      count: data.count,
      message: data.message,
    });
  } catch (error) {
    console.error("Error communicating with FastAPI backend:", error);
    return NextResponse.json(
      {
        error: "Failed to connect to backend service",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
