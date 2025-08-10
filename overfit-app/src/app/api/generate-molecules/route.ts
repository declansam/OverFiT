import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    // Parse request body for any parameters
    let body = {};
    try {
      body = await request.json();
    } catch {
      // Use default parameters if no body provided
      body = {
        num_samples: 100,
        temperature: 1.0,
        edge_threshold: 0.5,
      };
    }

    // Forward request to FastAPI backend
    const response = await fetch(`${FASTAPI_BASE_URL}/api/generate-molecules`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || "Failed to generate molecules" },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: data.success,
      message: data.message,
      molecules_count: data.molecules_count,
      validity_rate: data.validity_rate,
      uniqueness_rate: data.uniqueness_rate,
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
