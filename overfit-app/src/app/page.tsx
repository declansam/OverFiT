"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

// Simple SVG icons as React components
const FlaskIcon = () => (
  <svg
    width="24"
    height="24"
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

const SparklesIcon = () => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    <path d="M5 3v4" />
    <path d="M19 17v4" />
    <path d="M3 5h4" />
    <path d="M17 19h4" />
  </svg>
);

const SearchIcon = () => (
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
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.35-4.35" />
  </svg>
);

export default function Home() {
  const [smiles, setSmiles] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "healthy" | "unhealthy"
  >("checking");
  const router = useRouter();

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch("/api/health");
      if (response.ok) {
        setBackendStatus("healthy");
      } else {
        setBackendStatus("unhealthy");
      }
    } catch (error) {
      setBackendStatus("unhealthy");
    }
  };

  const handleGenerateMolecules = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch("/api/generate-molecules", {
        method: "POST",
      });

      if (response.ok) {
        // Navigate to molecules page after generation
        router.push("/molecules");
      } else {
        alert("Failed to generate molecules. Please try again.");
      }
    } catch (error) {
      console.error("Error generating molecules:", error);
      alert("An error occurred while generating molecules.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCheckHIV = async () => {
    if (!smiles.trim()) {
      alert("Please enter a SMILES string");
      return;
    }

    setIsChecking(true);
    try {
      const response = await fetch("/api/predict-hiv", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ smiles: smiles.trim() }),
      });

      const result = await response.json();

      if (response.ok) {
        // Show prediction result
        const activity = result.prediction === 1 ? "Active" : "Inactive";
        const confidence = (result.probability * 100).toFixed(1);
        alert(
          `HIV Inhibition Prediction:\n${activity} (${confidence}% confidence)`
        );
      } else {
        alert(result.error || "Failed to predict HIV inhibition");
      }
    } catch (error) {
      console.error("Error predicting HIV inhibition:", error);
      alert("An error occurred while predicting HIV inhibition.");
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col gap-8">
      {/* Header */}
      <header className="glass-subtle mx-8 my-2 mb-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center justify-center gap-4">
            <FlaskIcon />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              OverFiT
            </h1>
          </div>
          <div className="flex items-center gap-8">
            <div className="text-base text-gray-300 font-medium">
              Molecular Generation & Analysis
            </div>
            <div className="flex items-center gap-3">
              <div
                className={`w-3 h-3 rounded-full ${
                  backendStatus === "healthy"
                    ? "bg-green-400"
                    : backendStatus === "unhealthy"
                    ? "bg-red-400"
                    : "bg-yellow-400"
                }`}
              ></div>
              <span className="text-sm text-gray-400 font-medium">
                {backendStatus === "healthy"
                  ? "Backend Connected"
                  : backendStatus === "unhealthy"
                  ? "Backend Offline"
                  : "Checking..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center px-8 pb-8">
        <div className="max-w-7xl mx-auto flex items-center justify-center min-h-[calc(100vh-160px)]">
          <div className="w-full flex flex-col items-center justify-center gap-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
              {/* Left Side - Generate Molecules */}
              <div className="card-lg fade-in">
                <div className="text-center">
                  <div className="mb-8">
                    <div className="w-24 h-24 mx-auto mb-6 glass rounded-full flex items-center justify-center">
                      <SparklesIcon />
                    </div>
                    <h2 className="text-3xl font-bold mb-6">
                      Generate Molecules
                    </h2>
                    <p className="text-gray-300 text-lg leading-relaxed">
                      Use our VAE model to generate novel molecular structures
                    </p>
                  </div>

                  <div className="flex flex-col gap-8">
                    <div className="glass-subtle rounded-2xl">
                      <h3 className="text-xl font-semibold mb-4">
                        GraphVAE Model
                      </h3>
                      <p className="text-gray-300 text-sm mb-6 leading-relaxed">
                        Our Variational Autoencoder generates drug-like
                        molecules based on learned patterns from HIV-active
                        compounds. The model creates diverse molecular
                        structures with promising pharmaceutical properties.
                      </p>
                      <div className="flex items-center justify-center gap-6 text-sm text-gray-300">
                        <span>🧬 Molecular Diversity</span>
                        <span>⚗️ Drug-like Properties</span>
                        <span>🎯 HIV-focused Training</span>
                      </div>
                    </div>

                    <button
                      onClick={handleGenerateMolecules}
                      disabled={isGenerating}
                      className="btn-primary w-full py-4 text-lg font-semibold flex items-center justify-center gap-3"
                    >
                      {isGenerating ? (
                        <>
                          <div className="spinner"></div>
                          <span>Generating...</span>
                        </>
                      ) : (
                        <>
                          <SparklesIcon />
                          <span>Generate Molecules</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Right Side - HIV Inhibition Check */}
              <div
                className="card-lg fade-in"
                style={{ animationDelay: "0.2s" }}
              >
                <div className="text-center">
                  <div className="mb-8">
                    <div className="w-24 h-24 mx-auto mb-6 glass rounded-full flex items-center justify-center">
                      <SearchIcon />
                    </div>
                    <h2 className="text-3xl font-bold mb-6">
                      Check HIV Inhibition
                    </h2>
                    <p className="text-gray-300 text-lg leading-relaxed">
                      Predict HIV inhibition activity for any molecular
                      structure
                    </p>
                  </div>

                  <div className="flex flex-col gap-8">
                    <div className="glass-subtle rounded-2xl">
                      <h3 className="text-xl font-semibold mb-4">
                        SMILES Input
                      </h3>
                      <p className="text-gray-300 text-sm mb-6 leading-relaxed">
                        Enter a SMILES (Simplified Molecular Input Line Entry
                        System) string to predict the molecule's potential for
                        HIV inhibition activity.
                      </p>

                      <div className="mb-6">
                        <label
                          htmlFor="smiles-input"
                          className="block text-sm font-medium text-gray-300 mb-3"
                        >
                          SMILES String
                        </label>
                        <input
                          id="smiles-input"
                          type="text"
                          value={smiles}
                          onChange={(e) => setSmiles(e.target.value)}
                          placeholder="e.g., CCCCCCCCCc1ccc(C(=O)O)cc1"
                          className="input-glass"
                        />
                      </div>

                      <div className="text-sm text-gray-400">
                        Example: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O (Ibuprofen)
                      </div>
                    </div>

                    <button
                      onClick={handleCheckHIV}
                      disabled={isChecking || !smiles.trim()}
                      className={`w-full py-4 text-lg font-semibold flex items-center justify-center gap-3 ${
                        !smiles.trim()
                          ? "btn-secondary opacity-50 cursor-not-allowed"
                          : "btn-primary"
                      }`}
                    >
                      {isChecking ? (
                        <>
                          <div className="spinner"></div>
                          <span>Analyzing...</span>
                        </>
                      ) : (
                        <>
                          <SearchIcon />
                          <span>Check for HIV Inhibition</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Info Card */}
            <div
              className="glass-subtle rounded-2xl fade-in max-w-6xl mx-auto"
              style={{ animationDelay: "0.4s" }}
            >
              <div className="text-center">
                <h3 className="text-2xl font-semibold mb-8">How It Works</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                  <div>
                    <div className="text-blue-400 font-semibold mb-4 text-lg">
                      1. Molecule Generation
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Our GraphVAE model generates novel molecular structures by
                      learning from HIV-active compounds in the training
                      dataset.
                    </p>
                  </div>
                  <div>
                    <div className="text-purple-400 font-semibold mb-4 text-lg">
                      2. Structure Analysis
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Each generated molecule is analyzed for drug-like
                      properties, validity, and structural characteristics using
                      RDKit.
                    </p>
                  </div>
                  <div>
                    <div className="text-green-400 font-semibold mb-4 text-lg">
                      3. Activity Prediction
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Machine learning models predict the likelihood of HIV
                      inhibition activity based on molecular descriptors and
                      features.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
