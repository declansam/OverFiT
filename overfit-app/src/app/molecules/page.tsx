"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Molecule3DViewer from "../../components/Molecule3DViewer";

// Icons
const ArrowLeftIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m12 19-7-7 7-7" />
    <path d="M19 12H5" />
  </svg>
);

const FlaskIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M14 2v6.5l4.5 4.5c.4.4.5 1 .1 1.5L15 18H9l-3.6-3.5c-.4-.5-.3-1.1.1-1.5L10 8.5V2" />
    <line x1="10" y1="2" x2="14" y2="2" />
  </svg>
);

const ActivityIcon = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

const CopyIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
  </svg>
);

interface MoleculeData {
  smiles: string;
  index: number;
  properties?: {
    molecular_weight: number;
    logp: number;
    h_bond_donors: number;
    h_bond_acceptors: number;
    drug_like: boolean;
  };
}

export default function MoleculesPage() {
  const [molecules, setMolecules] = useState<MoleculeData[]>([]);
  const [uniqueMolecules, setUniqueMolecules] = useState<MoleculeData[]>([]);
  const [selectedMolecule, setSelectedMolecule] = useState<MoleculeData | null>(
    null
  );
  const [moleculeImage, setMoleculeImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const router = useRouter();

  // Handle cleanup on page navigation
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Clear any pending states that might cause issues
      setSelectedMolecule(null);
      setMoleculeImage(null);
      setPredictionResult(null);
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  useEffect(() => {
    loadMolecules();
  }, []);

  useEffect(() => {
    if (selectedMolecule) {
      loadMoleculeImage(selectedMolecule.smiles);
    }
  }, [selectedMolecule]);

  const loadMolecules = async () => {
    try {
      const response = await fetch("/api/get-molecules");
      if (response.ok) {
        const data = await response.json();
        // Handle both old and new response formats
        const moleculesList = data.molecules || [];
        setMolecules(moleculesList);

        // Filter for unique SMILES strings
        const uniqueMap = new Map<string, MoleculeData>();
        moleculesList.forEach((molecule: MoleculeData) => {
          if (!uniqueMap.has(molecule.smiles)) {
            uniqueMap.set(molecule.smiles, molecule);
          }
        });
        const uniqueList = Array.from(uniqueMap.values());
        setUniqueMolecules(uniqueList);

        if (uniqueList.length > 0) {
          setSelectedMolecule(uniqueList[0]);
        }
      } else {
        const errorData = await response.json();
        console.error("Failed to load molecules:", errorData.error);
      }
    } catch (error) {
      console.error("Error loading molecules:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMoleculeImage = async (smiles: string) => {
    setIsLoadingImage(true);
    try {
      const response = await fetch("/api/visualize-molecule", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ smiles }),
      });

      if (response.ok) {
        const data = await response.json();
        setMoleculeImage(data.image);
      } else {
        console.error("Failed to load molecule image");
        setMoleculeImage(null);
      }
    } catch (error) {
      console.error("Error loading molecule image:", error);
      setMoleculeImage(null);
    } finally {
      setIsLoadingImage(false);
    }
  };

  const handlePredictActivity = async () => {
    if (!selectedMolecule) return;

    setIsPredicting(true);
    setPredictionResult(null);

    try {
      const response = await fetch("/api/predict-hiv", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ smiles: selectedMolecule.smiles }),
      });

      const result = await response.json();

      if (response.ok) {
        setPredictionResult(result);
      } else {
        alert(result.error || "Failed to predict HIV inhibition");
      }
    } catch (error) {
      console.error("Error predicting HIV inhibition:", error);
      alert("An error occurred while predicting HIV inhibition.");
    } finally {
      setIsPredicting(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add a toast notification here
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="spinner mx-auto mb-6"></div>
          <p className="text-gray-300 text-base">Loading molecules...</p>
        </div>
      </div>
    );
  }

  if (uniqueMolecules.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center card-lg max-w-md">
          <p className="text-gray-300 text-base mb-6">
            No molecules generated yet.
          </p>
          <button
            onClick={() => {
              // Clear states before navigation
              setSelectedMolecule(null);
              setMoleculeImage(null);
              setPredictionResult(null);
              router.push("/");
            }}
            className="btn-primary"
          >
            Generate Molecules
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="glass-subtle mx-8 mt-8 mb-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <button
              onClick={() => {
                // Clear molecule viewer state before navigation
                setSelectedMolecule(null);
                setMoleculeImage(null);
                setPredictionResult(null);
                router.push("/");
              }}
              className="btn-secondary p-4"
            >
              <ArrowLeftIcon />
            </button>
            <FlaskIcon />
            <h1 className="text-3xl font-bold">Generated Molecules</h1>
          </div>
          <div className="text-base text-gray-300 font-medium">
            {uniqueMolecules.length} unique molecules generated
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        <div className="mx-8">
          <div className="flex gap-8">
            {/* Left Sidebar - SMILES List */}
            <div className="w-80">
              <div className="flex flex-col">
                <h2 className="text-xl font-semibold mb-4">SMILES Library</h2>
                <div className="flex-1 flex flex-col gap-4">
                  {uniqueMolecules.map((molecule, index) => (
                    <button
                      key={index}
                      onClick={() => setSelectedMolecule(molecule)}
                      className={`text-left rounded-sm transition-all border ${
                        selectedMolecule?.index === molecule.index
                          ? "bg-blue-600/20 border-blue-500 text-white"
                          : "bg-gray-800/50 border-gray-700 hover:bg-gray-700/50 hover:border-gray-600"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          #{index + 1}
                        </span>
                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(molecule.smiles);
                          }}
                          className="p-2 hover:bg-white/10 rounded cursor-pointer"
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              e.stopPropagation();
                              copyToClipboard(molecule.smiles);
                            }
                          }}
                        >
                          <CopyIcon />
                        </div>
                      </div>
                      <div className="text-xs text-gray-400 mt-2 truncate">
                        {molecule.smiles}
                      </div>
                      {molecule.properties?.drug_like && (
                        <div className="text-xs text-green-400 mt-2">
                          ✓ Drug-like
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col gap-8">
              {/* Molecule Info Header */}
              {selectedMolecule && (
                <div className="card gap-4">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-2xl font-semibold">
                      Molecule #{selectedMolecule.index + 1}
                    </h2>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={handlePredictActivity}
                        disabled={isPredicting || !selectedMolecule}
                        className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold ${
                          !selectedMolecule
                            ? "btn-secondary opacity-50 cursor-not-allowed"
                            : "btn-primary"
                        }`}
                      >
                        {isPredicting ? (
                          <>
                            <div className="spinner w-4 h-4"></div>
                            <span>Predicting...</span>
                          </>
                        ) : (
                          <>
                            <ActivityIcon />
                            <span>Predict HIV Inhibit</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => copyToClipboard(selectedMolecule.smiles)}
                        className="btn-secondary px-4 py-2"
                      >
                        Copy SMILES
                      </button>
                    </div>
                  </div>
                  <div className="glass-subtle p-4 rounded text-sm font-mono break-all">
                    {selectedMolecule.smiles}
                  </div>
                  {/* Prediction Result Display */}
                  {predictionResult && (
                    <div className="mt-4 glass-subtle p-4 rounded">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div
                            className={`text-lg font-bold ${
                              predictionResult.prediction === 1
                                ? "text-green-400"
                                : "text-red-400"
                            }`}
                          >
                            {predictionResult.prediction === 1
                              ? "HIV ACTIVE"
                              : "HIV INACTIVE"}
                          </div>
                          <div className="text-sm text-gray-300">
                            Confidence:{" "}
                            {(predictionResult.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        {predictionResult.descriptors && (
                          <div className="flex gap-6 text-xs">
                            <div className="text-center">
                              <div className="font-semibold text-blue-400">
                                {predictionResult.descriptors.tpsa?.toFixed(
                                  1
                                ) || "N/A"}
                              </div>
                              <div className="text-gray-400">TPSA</div>
                            </div>
                            <div className="text-center">
                              <div className="font-semibold text-green-400">
                                {predictionResult.descriptors.aromatic_rings ||
                                  "N/A"}
                              </div>
                              <div className="text-gray-400">
                                Aromatic Rings
                              </div>
                            </div>
                            <div className="text-center">
                              <div className="font-semibold text-purple-400">
                                {predictionResult.descriptors.rotatable_bonds ||
                                  "N/A"}
                              </div>
                              <div className="text-gray-400">
                                Rotatable Bonds
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Visualizations Side by Side */}
              <div className="flex gap-8 flex-1">
                {/* 2D Structure */}
                <div className="flex-1 card">
                  <h3 className="text-xl font-semibold mb-6">2D Structure</h3>
                  <div className="bg-white rounded-xl p-6 min-h-[500px] flex items-center justify-center">
                    {isLoadingImage ? (
                      <div className="text-center">
                        <div className="spinner mx-auto mb-3"></div>
                        <p className="text-gray-600 text-sm">
                          Generating structure...
                        </p>
                      </div>
                    ) : moleculeImage ? (
                      <img
                        src={`data:image/png;base64,${moleculeImage}`}
                        alt="Molecule structure"
                        className="max-w-full max-h-full object-contain"
                      />
                    ) : (
                      <p className="text-gray-600 text-sm">
                        Failed to generate structure
                      </p>
                    )}
                  </div>
                </div>

                {/* 3D Viewer */}
                <div className="flex-1 card">
                  <h3 className="text-xl font-semibold mb-6">3D Structure</h3>
                  <div className="min-h-[500px] flex items-center justify-center">
                    {selectedMolecule ? (
                      <Molecule3DViewer
                        smiles={selectedMolecule.smiles}
                        width={500}
                        height={500}
                        className="mx-auto"
                      />
                    ) : (
                      <p className="text-gray-400 text-sm">
                        Select a molecule to view 3D structure
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Molecular Properties */}
              {selectedMolecule?.properties && (
                <div className="card">
                  <h3 className="text-xl font-semibold mb-6">
                    Molecular Properties
                  </h3>
                  <div className="flex justify-center gap-12">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">
                        {selectedMolecule.properties.molecular_weight.toFixed(
                          1
                        )}
                      </div>
                      <div className="text-sm text-gray-400">
                        Molecular Weight
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">
                        {selectedMolecule.properties.logp.toFixed(2)}
                      </div>
                      <div className="text-sm text-gray-400">LogP</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400">
                        {selectedMolecule.properties.h_bond_donors}
                      </div>
                      <div className="text-sm text-gray-400">H-Bond Donors</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-400">
                        {selectedMolecule.properties.h_bond_acceptors}
                      </div>
                      <div className="text-sm text-gray-400">
                        H-Bond Acceptors
                      </div>
                    </div>
                  </div>
                  <div className="mt-6 text-center">
                    <span
                      className={`text-lg px-6 py-3 rounded-lg font-semibold ${
                        selectedMolecule.properties.drug_like
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {selectedMolecule.properties.drug_like
                        ? "✓ Drug-like"
                        : "✗ Non drug-like"}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
