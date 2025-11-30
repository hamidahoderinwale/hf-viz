import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config/api';
import LoadingProgress from '../components/ui/LoadingProgress';
import AdoptionCurve, { AdoptionDataPoint } from '../components/visualizations/AdoptionCurve';
import './FamiliesPage.css';

interface Family {
  family: string;
  count: number;
  root_model?: string;
  family_count?: number;  // Number of separate family trees
  root_models?: string[]; // List of root models for this org
}

interface AdoptionModel {
  model_id: string;
  downloads: number;
  created_at: string;
}

interface FamilyAdoptionData {
  family: string;
  data: AdoptionDataPoint[];
  color: string;
}

export default function FamiliesPage() {
  const [families, setFamilies] = useState<Family[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [adoptionData, setAdoptionData] = useState<AdoptionDataPoint[]>([]);
  const [loadingAdoption, setLoadingAdoption] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  
  // Comparison mode - enabled by default
  const [comparisonMode, setComparisonMode] = useState(true);
  const [selectedFamiliesForComparison, setSelectedFamiliesForComparison] = useState<Set<string>>(new Set());
  const [familyAdoptionData, setFamilyAdoptionData] = useState<FamilyAdoptionData[]>([]);
  const [loadingComparison, setLoadingComparison] = useState(false);

  useEffect(() => {
    const fetchFamilies = async () => {
      setLoading(true);
      setLoadingProgress(0);
      
      try {
        setLoadingProgress(20);
        
        // Fetch models data to count by organization
        const response = await fetch(`${API_BASE}/api/models?max_points=10000&format=json`);
        if (!response.ok) throw new Error('Failed to fetch models');
        
        setLoadingProgress(40);
        const data = await response.json();
        const models = Array.isArray(data) ? data : (data.models || []);
        
        setLoadingProgress(60);
        
        // Group by organization (model_id prefix)
        // Also track models by family_depth to show lineage distribution
        const familyMap = new Map<string, { total: number; byDepth: Map<number, number> }>();
        
        models.forEach((model: any) => {
          const org = model.model_id?.split('/')[0] || 'unknown';
          const depth = model.family_depth ?? 0;
          
          if (!familyMap.has(org)) {
            familyMap.set(org, { total: 0, byDepth: new Map() });
          }
          
          const orgData = familyMap.get(org)!;
          orgData.total += 1;
          orgData.byDepth.set(depth, (orgData.byDepth.get(depth) || 0) + 1);
        });
        
        setLoadingProgress(80);
        
        // Convert to array and calculate depth distribution info
        const familiesList: Family[] = Array.from(familyMap.entries())
          .map(([family, data]) => {
            // Count unique depths to show family tree complexity
            const depthCount = data.byDepth.size;
            const maxDepth = Math.max(...Array.from(data.byDepth.keys()));
            
            return {
              family,
              count: data.total,
              family_count: depthCount > 1 ? depthCount : undefined, // Number of depth levels
              root_models: maxDepth > 0 ? [`max depth: ${maxDepth}`] : undefined
            };
          })
          .sort((a, b) => b.count - a.count)
          .slice(0, 50);
        
        setFamilies(familiesList);
        
        // Initialize comparison mode with top 5 families (if not already set)
        if (familiesList.length >= 5 && selectedFamiliesForComparison.size === 0) {
          const top5 = familiesList.slice(0, 5).map(f => f.family);
          setSelectedFamiliesForComparison(new Set(top5));
        }
        
        setLoadingProgress(100);
        setLoading(false);
      } catch {
        setLoading(false);
      }
    };

    fetchFamilies();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch adoption data when family is selected
  useEffect(() => {
    if (!selectedFamily) {
      setAdoptionData([]);
      return;
    }

    const fetchAdoptionData = async () => {
      setLoadingAdoption(true);
      try {
        const response = await fetch(`${API_BASE}/api/family/adoption?family=${encodeURIComponent(selectedFamily)}&limit=200`);
        if (!response.ok) throw new Error('Failed to fetch adoption data');
        
        const data = await response.json();
        const models: AdoptionModel[] = data.models || [];
        
        // Transform to AdoptionDataPoint format
        const chartData: AdoptionDataPoint[] = models
          .filter((m) => m.created_at)
          .map((m) => ({
            date: new Date(m.created_at),
            downloads: m.downloads,
            modelId: m.model_id,
          }))
          .sort((a, b) => a.date.getTime() - b.date.getTime());

        setAdoptionData(chartData);
        
        // Select the model with highest downloads by default
        if (chartData.length > 0) {
          const topModel = chartData.reduce((max, current) => 
            current.downloads > max.downloads ? current : max
          );
          setSelectedModel(topModel.modelId);
        }
      } catch {
        setAdoptionData([]);
      } finally {
        setLoadingAdoption(false);
      }
    };

    fetchAdoptionData();
  }, [selectedFamily]);

  // Fetch adoption data for comparison mode
  useEffect(() => {
    if (!comparisonMode || selectedFamiliesForComparison.size === 0) {
      setFamilyAdoptionData([]);
      return;
    }

    const fetchComparisonData = async () => {
      setLoadingComparison(true);
      try {
        const familyNames = Array.from(selectedFamiliesForComparison);
        const FAMILY_COLORS = [
          '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
          '#ec4899', '#06b6d4', '#f97316'
        ];

        const adoptionPromises = familyNames.map(async (familyName, idx): Promise<FamilyAdoptionData | null> => {
          try {
            const response = await fetch(`${API_BASE}/api/family/adoption?family=${encodeURIComponent(familyName)}&limit=200`);
            if (!response.ok) throw new Error(`Failed to fetch adoption data for ${familyName}`);
            
            const data = await response.json();
            const models: AdoptionModel[] = data.models || [];
            
            const chartData: AdoptionDataPoint[] = models
              .filter((m) => m.created_at)
              .map((m) => ({
                date: new Date(m.created_at),
                downloads: m.downloads,
                modelId: m.model_id,
              }))
              .sort((a, b) => a.date.getTime() - b.date.getTime());

            return {
              family: familyName,
              data: chartData,
              color: FAMILY_COLORS[idx % FAMILY_COLORS.length],
            };
          } catch {
            return null;
          }
        });

        const results = await Promise.all(adoptionPromises);
        const validResults = results.filter((r): r is FamilyAdoptionData => r !== null);
        setFamilyAdoptionData(validResults);
      } catch {
        setFamilyAdoptionData([]);
      } finally {
        setLoadingComparison(false);
      }
    };

    fetchComparisonData();
  }, [comparisonMode, selectedFamiliesForComparison]);

  if (loading) {
    return (
      <LoadingProgress 
        message="Loading families..." 
        progress={loadingProgress}
        subMessage="Analyzing model families"
      />
    );
  }

  return (
    <div className="families-page">
      <div className="page-header">
        <h1>Model Families</h1>
        <p className="page-description">
          Explore the largest model families on Hugging Face, organized by organization and model lineage.
        </p>
      </div>

      {/* Adoption Curve Section - Always visible */}
      <section className="adoption-section">
          <div className="adoption-header">
            <h2>
              Adoption Curve
            </h2>
            <div className="adoption-controls">
              <button
                className="comparison-toggle-btn"
                onClick={() => {
                  setComparisonMode(!comparisonMode);
                  if (!comparisonMode) {
                    setSelectedFamily(null);
                    setAdoptionData([]);
                  }
                }}
              >
                {comparisonMode ? 'Single View' : 'Compare Top 5'}
              </button>
            </div>
          </div>
          
          {comparisonMode ? (
            <>
              {/* Family selection checkboxes */}
              <div className="family-selection">
                <p className="selection-label">Select families to compare:</p>
                <div className="family-checkboxes">
                  {families.slice(0, 10).map((family) => (
                    <label key={family.family} className="family-checkbox">
                      <input
                        type="checkbox"
                        checked={selectedFamiliesForComparison.has(family.family)}
                        onChange={(e) => {
                          const newSet = new Set(selectedFamiliesForComparison);
                          if (e.target.checked) {
                            newSet.add(family.family);
                          } else {
                            newSet.delete(family.family);
                          }
                          setSelectedFamiliesForComparison(newSet);
                        }}
                      />
                      <span>
                        {family.family} ({family.count.toLocaleString()} models)
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {loadingComparison ? (
                <LoadingProgress 
                  message="Loading comparison data..." 
                  progress={0}
                  subMessage="Fetching adoption data for selected families"
                />
              ) : familyAdoptionData.length > 0 ? (
                <div className="adoption-curve-wrapper">
                  <div className="chart-info">
                    <p className="chart-subtitle">
                      {families
                        .filter(f => selectedFamiliesForComparison.has(f.family))
                        .reduce((sum, f) => sum + f.count, 0)
                        .toLocaleString()} models across selected organizations • Cumulative downloads over time
                    </p>
                  </div>
                  <AdoptionCurve
                    data={[]}
                    families={familyAdoptionData}
                    width={Math.min(1000, window.innerWidth - 100)}
                    height={400}
                  />
                </div>
              ) : (
                <div className="adoption-empty">
                  <p>No adoption data available for selected families.</p>
                </div>
              )}
            </>
          ) : (
            <>
              {loadingAdoption ? (
                <LoadingProgress 
                  message="Loading adoption data..." 
                  progress={0}
                  subMessage="Fetching model history"
                />
              ) : adoptionData.length > 0 ? (
                <div className="adoption-curve-wrapper">
                  <div className="chart-info">
                    <p className="chart-subtitle">
                      {adoptionData.length} models • Cumulative downloads over time
                    </p>
                  </div>
                  <AdoptionCurve
                    data={adoptionData}
                    selectedModel={selectedModel}
                    width={Math.min(1000, window.innerWidth - 100)}
                    height={400}
                  />
                </div>
              ) : (
                <div className="adoption-empty">
                  <p>No adoption data available for this family.</p>
                </div>
              )}
            </>
          )}
      </section>

      <section className="families-section">
        <h2>Top Families by Model Count</h2>
        <div className="families-list">
          {families.map((family, idx) => (
            <div 
              key={family.family} 
              className={`family-item ${selectedFamily === family.family ? 'selected' : ''}`}
              onClick={() => {
                if (selectedFamily === family.family) {
                  setSelectedFamily(null);
                  setAdoptionData([]);
                  setSelectedModel(undefined);
                } else {
                  setSelectedFamily(family.family);
                }
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  if (selectedFamily === family.family) {
                    setSelectedFamily(null);
                    setAdoptionData([]);
                    setSelectedModel(undefined);
                  } else {
                    setSelectedFamily(family.family);
                  }
                }
              }}
            >
              <div className="family-info">
                <span className="rank">{idx + 1}.</span>
                <div className="family-details">
                  <span className="family-name">{family.family}</span>
                  <span className="family-count">
                    {family.count.toLocaleString()} models
                  </span>
                </div>
              </div>
              <div className="family-stats">
                <div className="stat-badge">
                  {((family.count / families.reduce((sum, f) => sum + f.count, 0)) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
