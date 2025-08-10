"use client";

import { useEffect, useRef, useState } from "react";

// Define types for the 3Dmol library
declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface Molecule3DData {
  smiles: string;
  pdb?: string;
  sdf?: string;
  xyz?: string;
  num_atoms: number;
  optimized: boolean;
  force_field?: string;
}

interface Molecule3DViewerProps {
  smiles: string;
  className?: string;
  width?: number;
  height?: number;
  colorScheme?: "element" | "cpk" | "amino" | "rainbow" | "hydrophobicity";
  style?: "stick" | "sphere" | "line" | "cross";
}

export default function Molecule3DViewer({
  smiles,
  className = "",
  width = 400,
  height = 400,
  colorScheme = "element",
  style = "stick",
}: Molecule3DViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [moleculeData, setMoleculeData] = useState<Molecule3DData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [is3DmolLoaded, setIs3DmolLoaded] = useState(false);
  const [currentColorScheme, setCurrentColorScheme] = useState(colorScheme);
  const [currentStyle, setCurrentStyle] = useState(style);
  const viewerInstance = useRef<any>(null);

  // Load 3Dmol.js library
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://3dmol.csb.pitt.edu/build/3Dmol-min.js";
    script.onload = () => {
      setIs3DmolLoaded(true);
    };
    script.onerror = () => {
      setError("Failed to load 3Dmol.js library");
    };
    document.head.appendChild(script);

    return () => {
      if (document.head.contains(script)) {
        document.head.removeChild(script);
      }
    };
  }, []);

  // Load 3D molecule data
  useEffect(() => {
    if (!smiles || !is3DmolLoaded) return;

    const load3DMolecule = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch("/api/generate-3d-molecule", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            smiles,
            optimize: true,
            force_field: "MMFF",
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to generate 3D structure");
        }

        const data = await response.json();

        if (data.success) {
          setMoleculeData(data);
        } else {
          setError(data.error || "Failed to generate 3D structure");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsLoading(false);
      }
    };

    load3DMolecule();
  }, [smiles, is3DmolLoaded]);

  // Helper function to get explicit color definitions
  const getExplicitColors = (scheme: string, viewer: any) => {
    switch (scheme) {
      case "element":
        return {
          C: 0x909090, // Gray for carbon
          N: 0x3050f8, // Blue for nitrogen
          O: 0xff0d0d, // Red for oxygen
          S: 0xffff30, // Yellow for sulfur
          P: 0xff8000, // Orange for phosphorus
          F: 0x90e050, // Light green for fluorine
          Cl: 0x1ff01f, // Green for chlorine
          Br: 0xa62929, // Brown for bromine
          I: 0x940094, // Purple for iodine
          H: 0xffffff, // White for hydrogen
        };
      case "cpk":
        return {
          C: 0x000000, // Black for carbon
          N: 0x8f8fff, // Light blue for nitrogen
          O: 0xff0000, // Red for oxygen
          S: 0xffc832, // Yellow for sulfur
          P: 0xffa500, // Orange for phosphorus
          F: 0x00ff00, // Green for fluorine
          Cl: 0x00ff00, // Green for chlorine
          Br: 0xa52a2a, // Brown for bromine
          I: 0x9932cc, // Purple for iodine
          H: 0xffffff, // White for hydrogen
        };
      case "amino":
        return {
          C: 0xc8c8c8, // Light gray for carbon
          N: 0x8080ff, // Light blue for nitrogen
          O: 0xff8080, // Light red for oxygen
          S: 0xffd700, // Gold for sulfur
          P: 0xffa500, // Orange for phosphorus
          F: 0x80ff80, // Light green for fluorine
          Cl: 0x80ff80, // Light green for chlorine
          Br: 0xd2691e, // Brown for bromine
          I: 0xda70d6, // Orchid for iodine
          H: 0xffffff, // White for hydrogen
        };
      case "hydrophobicity":
        return {
          C: 0x00ff00, // Green for hydrophobic carbon
          N: 0x0000ff, // Blue for hydrophilic nitrogen
          O: 0xff0000, // Red for hydrophilic oxygen
          S: 0xffff00, // Yellow for sulfur
          P: 0xff8000, // Orange for phosphorus
          F: 0x00ffff, // Cyan for fluorine
          Cl: 0x00ffff, // Cyan for chlorine
          Br: 0x00ffff, // Cyan for bromine
          I: 0x00ffff, // Cyan for iodine
          H: 0xffffff, // White for hydrogen
        };

      case "rainbow":
        // Generate rainbow colors based on atom index
        return null; // Will handle this differently
      default:
        return null;
    }
  };

  // Helper function to apply rainbow coloring
  const applyRainbowColoring = (viewer: any, numAtoms: number) => {
    for (let i = 0; i < numAtoms; i++) {
      const hue = (i / numAtoms) * 360;
      const saturation = 80 + (i % 20); // 80-100% saturation for vibrant colors
      const lightness = 40 + (i % 30); // 40-70% lightness for good contrast
      const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

      // Apply to specific atom by index
      switch (currentStyle) {
        case "stick":
          viewer.setStyle(
            { serial: i + 1 },
            {
              stick: { radius: 0.15, color: color },
              sphere: { scale: 0.25, color: color },
            }
          );
          break;
        case "sphere":
          viewer.setStyle(
            { serial: i + 1 },
            {
              sphere: { scale: 0.4, color: color },
            }
          );
          break;
        case "line":
          viewer.setStyle(
            { serial: i + 1 },
            {
              line: { linewidth: 2, color: color },
            }
          );
          break;
        case "cross":
          viewer.setStyle(
            { serial: i + 1 },
            {
              cross: { linewidth: 2, color: color },
            }
          );
          break;
        default:
          viewer.setStyle(
            { serial: i + 1 },
            {
              stick: { radius: 0.15, color: color },
              sphere: { scale: 0.25, color: color },
            }
          );
      }
    }
  };

  // Helper function to apply element-specific coloring
  const applyElementColoring = (
    viewer: any,
    styleType: string,
    colorScheme: string
  ) => {
    const colors = getExplicitColors(colorScheme, viewer);

    if (!colors) {
      // Fallback to basic coloring for unsupported schemes
      const basicStyle = getBasicStyle(styleType);
      viewer.setStyle({}, basicStyle);
      return;
    }

    // Apply colors for each element type
    Object.entries(colors).forEach(([element, color]) => {
      const selector = { elem: element };

      switch (styleType) {
        case "stick":
          viewer.setStyle(selector, {
            stick: { radius: 0.15, color: color },
            sphere: { scale: 0.25, color: color },
          });
          break;
        case "sphere":
          viewer.setStyle(selector, {
            sphere: { scale: 0.4, color: color },
          });
          break;
        case "line":
          viewer.setStyle(selector, {
            line: { linewidth: 2, color: color },
          });
          break;
        case "cross":
          viewer.setStyle(selector, {
            cross: { linewidth: 2, color: color },
          });
          break;

        default:
          viewer.setStyle(selector, {
            stick: { radius: 0.15, color: color },
            sphere: { scale: 0.25, color: color },
          });
      }
    });
  };

  // Helper function to get basic style without specific coloring
  const getBasicStyle = (styleType: string) => {
    switch (styleType) {
      case "stick":
        return {
          stick: { radius: 0.15 },
          sphere: { scale: 0.25 },
        };
      case "sphere":
        return {
          sphere: { scale: 0.4 },
        };
      case "line":
        return {
          line: { linewidth: 2 },
        };
      case "cross":
        return {
          cross: { linewidth: 2 },
        };

      default:
        return {
          stick: { radius: 0.15 },
          sphere: { scale: 0.25 },
        };
    }
  };

  // Initialize 3D viewer
  useEffect(() => {
    if (!moleculeData || !is3DmolLoaded || !viewerRef.current) return;

    // Clean up previous viewer
    if (viewerInstance.current) {
      viewerInstance.current.clear();
    }

    try {
      // Create viewer
      const viewer = window.$3Dmol.createViewer(viewerRef.current, {
        width,
        height,
        backgroundColor: "white",
      });

      // Add molecule data - prefer SDF, then PDB, then XYZ
      let moleculeAdded = false;

      if (moleculeData.sdf) {
        viewer.addModel(moleculeData.sdf, "sdf");
        moleculeAdded = true;
      } else if (moleculeData.pdb) {
        viewer.addModel(moleculeData.pdb, "pdb");
        moleculeAdded = true;
      } else if (moleculeData.xyz) {
        viewer.addModel(moleculeData.xyz, "xyz");
        moleculeAdded = true;
      }

      if (moleculeAdded) {
        console.log(
          `Applying color scheme: ${currentColorScheme}, style: ${currentStyle}`
        );

        // Apply styling and coloring based on the selected scheme
        if (currentColorScheme === "rainbow") {
          // Apply rainbow coloring
          applyRainbowColoring(viewer, moleculeData.num_atoms);
        } else {
          // Apply element-specific coloring
          applyElementColoring(viewer, currentStyle, currentColorScheme);
        }

        // Zoom to fit
        viewer.zoomTo();

        // Render
        viewer.render();

        // Store viewer instance
        viewerInstance.current = viewer;
      } else {
        setError("No valid 3D structure data available");
      }
    } catch (err) {
      setError("Failed to initialize 3D viewer");
    }
  }, [
    moleculeData,
    is3DmolLoaded,
    width,
    height,
    currentColorScheme,
    currentStyle,
  ]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (viewerInstance.current) {
        viewerInstance.current.clear();
      }
    };
  }, []);

  const formatInfo = () => {
    if (!moleculeData) return null;

    return (
      <div className="text-xs text-gray-400 mt-4 space-y-2">
        <div>Atoms: {moleculeData.num_atoms}</div>
        <div>
          {moleculeData.optimized ? "✓ Optimized" : "Not optimized"}
          {moleculeData.force_field && ` (${moleculeData.force_field})`}
        </div>
        <div>
          Format: {moleculeData.sdf ? "SDF" : moleculeData.pdb ? "PDB" : "XYZ"}
        </div>
      </div>
    );
  };

  return (
    <div className={`molecule-3d-viewer ${className} space-y-4`}>
      <div className="mb-6">
        <p className="text-sm text-gray-400">
          Interactive 3D molecular visualization
        </p>
      </div>

      <div
        className="border border-gray-300 rounded-lg overflow-hidden bg-white relative"
        style={{ width, height }}
      >
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/90 z-10">
            <div className="text-center">
              <div className="spinner mx-auto mb-3"></div>
              <p className="text-gray-600 text-sm">
                Generating 3D structure...
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white">
            <div className="text-center p-6">
              <p className="text-red-600 text-sm">{error}</p>
              <button
                onClick={() => window.location.reload()}
                className="mt-3 text-sm text-blue-600 hover:text-blue-800 px-3 py-2 rounded-lg bg-blue-50 hover:bg-blue-100 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        <div
          ref={viewerRef}
          style={{ width: "100%", height: "100%" }}
          className={error || isLoading ? "invisible" : ""}
        />
      </div>

      {formatInfo()}

      {/* Controls */}
      <div className="mt-8 glass-subtle space-y-4">
        <h4 className="text-lg font-semibold mb-6 text-center">
          Visualization Controls
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Color Scheme Selector */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Color Scheme
            </label>
            <div className="relative">
              <select
                value={currentColorScheme}
                onChange={(e) => setCurrentColorScheme(e.target.value as any)}
                className="w-full px-4 py-3 text-sm bg-white/5 border border-white/10 rounded-xl focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-400/20 transition-all backdrop-blur-md text-white appearance-none cursor-pointer hover:bg-white/10"
              >
                <option value="element" className="bg-gray-800 text-white">
                  Element (Jmol)
                </option>
                <option value="cpk" className="bg-gray-800 text-white">
                  CPK (Rasmol)
                </option>
                <option value="rainbow" className="bg-gray-800 text-white">
                  Rainbow
                </option>
                <option value="amino" className="bg-gray-800 text-white">
                  Amino Acid
                </option>
                <option
                  value="hydrophobicity"
                  className="bg-gray-800 text-white"
                >
                  Hydrophobicity
                </option>
              </select>
              {/* Custom dropdown arrow */}
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <svg
                  className="h-4 w-4 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
            </div>
          </div>

          {/* Style Selector */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Representation
            </label>
            <div className="relative">
              <select
                value={currentStyle}
                onChange={(e) => setCurrentStyle(e.target.value as any)}
                className="w-full px-4 py-3 text-sm bg-white/5 border border-white/10 rounded-xl focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-400/20 transition-all backdrop-blur-md text-white appearance-none cursor-pointer hover:bg-white/10"
              >
                <option value="stick" className="bg-gray-800 text-white">
                  Ball & Stick
                </option>
                <option value="sphere" className="bg-gray-800 text-white">
                  Space Filling
                </option>
                <option value="line" className="bg-gray-800 text-white">
                  Wireframe
                </option>
                <option value="cross" className="bg-gray-800 text-white">
                  Cross
                </option>
              </select>
              {/* Custom dropdown arrow */}
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <svg
                  className="h-4 w-4 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Color Buttons */}
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-gray-300 whitespace-nowrap">
              Quick Select:
            </span>
            <div className="h-px flex-1 bg-gradient-to-r from-white/20 to-transparent"></div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {[
              {
                key: "element",
                label: "Element",
                color: "from-blue-500 to-blue-600",
                icon: "⚛️",
              },
              {
                key: "cpk",
                label: "CPK",
                color: "from-green-500 to-green-600",
                icon: "🟢",
              },
              {
                key: "rainbow",
                label: "Rainbow",
                color: "from-pink-500 via-purple-500 to-indigo-500",
                icon: "🌈",
              },
              {
                key: "amino",
                label: "Amino",
                color: "from-teal-500 to-teal-600",
                icon: "🧬",
              },
              {
                key: "hydrophobicity",
                label: "Hydrophobic",
                color: "from-cyan-500 to-cyan-600",
                icon: "💧",
              },
            ].map(({ key, label, color, icon }) => (
              <button
                key={key}
                onClick={() => setCurrentColorScheme(key as any)}
                className={`relative px-4 py-3 text-sm font-medium rounded-xl transition-all duration-300 transform hover:scale-105 focus:scale-105 focus:outline-none focus:ring-2 focus:ring-white/20 ${
                  currentColorScheme === key
                    ? `bg-gradient-to-r ${color} text-white shadow-lg ring-2 ring-white/30`
                    : `bg-white/5 text-gray-300 hover:bg-white/10 border border-white/10 hover:border-white/20`
                }`}
              >
                <div className="flex flex-col items-center gap-1">
                  <span className="text-lg">{icon}</span>
                  <span className="text-xs">{label}</span>
                </div>
                {currentColorScheme === key && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-white rounded-full animate-pulse"></div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Style Quick Buttons */}
        <div className="mt-6 space-y-4">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-gray-300 whitespace-nowrap">
              Representation:
            </span>
            <div className="h-px flex-1 bg-gradient-to-r from-white/20 to-transparent"></div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {[
              { key: "stick", label: "Ball & Stick", icon: "🔗" },
              { key: "sphere", label: "Space Filling", icon: "⚪" },
              { key: "line", label: "Wireframe", icon: "📐" },
              { key: "cross", label: "Cross", icon: "✚" },
            ].map(({ key, label, icon }) => (
              <button
                key={key}
                onClick={() => setCurrentStyle(key as any)}
                className={`relative px-4 py-3 text-sm font-medium rounded-xl transition-all duration-300 transform hover:scale-105 focus:scale-105 focus:outline-none focus:ring-2 focus:ring-white/20 ${
                  currentStyle === key
                    ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-lg ring-2 ring-white/30"
                    : "bg-white/5 text-gray-300 hover:bg-white/10 border border-white/10 hover:border-white/20"
                }`}
              >
                <div className="flex flex-col items-center gap-1">
                  <span className="text-lg">{icon}</span>
                  <span className="text-xs">{label}</span>
                </div>
                {currentStyle === key && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-white rounded-full animate-pulse"></div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-6 glass-subtle bg-white/5 border border-white/10">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
          <h5 className="text-sm font-medium text-gray-300">
            Interaction Guide
          </h5>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-gray-400">
          <div className="flex items-center gap-2">
            <span className="text-blue-400">🖱️</span>
            <span>Drag to rotate</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-400">🔄</span>
            <span>Scroll to zoom</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-orange-400">👆</span>
            <span>Click & drag to pan</span>
          </div>
        </div>
      </div>
    </div>
  );
}
