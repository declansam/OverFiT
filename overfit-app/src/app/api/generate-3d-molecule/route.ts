import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const {
      smiles,
      optimize = true,
      force_field = "MMFF",
    } = await request.json();

    if (!smiles) {
      return NextResponse.json(
        { error: "SMILES string is required" },
        { status: 400 }
      );
    }

    // Forward request to FastAPI backend
    const response = await fetch(`${FASTAPI_BASE_URL}/api/generate-3d-molecule`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        smiles,
        optimize,
        force_field,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || "Failed to generate 3D molecule" },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: data.success,
      smiles: data.smiles,
      pdb: data.pdb,
      sdf: data.sdf,
      xyz: data.xyz,
      num_atoms: data.num_atoms,
      optimized: data.optimized,
      force_field: data.force_field,
      error: data.error,
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
