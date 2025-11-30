/**
 * Intro Modal - Brief onboarding guide for the dashboard
 */
import React, { useState, useEffect } from 'react';
import { X, Palette, Maximize2, Search, Move3D, Sparkles } from 'lucide-react';
import './IntroModal.css';

interface IntroModalProps {
  onClose: () => void;
}

export default function IntroModal({ onClose }: IntroModalProps) {
  const [dontShowAgain, setDontShowAgain] = useState(false);

  const handleClose = () => {
    if (dontShowAgain) {
      localStorage.setItem('hf-intro-dismissed', 'true');
    }
    onClose();
  };

  // Close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [dontShowAgain]);

  return (
    <div className="intro-modal">
      <div className="intro-header">
        <span className="intro-title">Quick Guide</span>
        <button className="intro-close" onClick={handleClose} title="Close">
          <X size={14} />
        </button>
      </div>

      <div className="intro-content">
        <p className="intro-desc">
          Visualizing <strong>2M+ ML models</strong> from Hugging Face by text embeddings.
          <br />
          <span className="intro-desc-sub">Each point represents a model positioned by semantic similarity.</span>
        </p>

        <div className="intro-section">
          <div className="intro-section-title">
            <Palette size={12} />
            Colors
          </div>
          <ul className="intro-list">
            <li title="Colors based on how many generations a model is from its root parent">
              <span className="intro-color family"></span>
              <strong>Family</strong>
              <span className="intro-detail">Lineage depth</span>
            </li>
            <li title="Colors based on which ML framework/library the model uses">
              <span className="intro-color library"></span>
              <strong>Library</strong>
              <span className="intro-detail">ML framework</span>
            </li>
            <li title="Colors based on what the model does">
              <span className="intro-color task"></span>
              <strong>Task</strong>
              <span className="intro-detail">Model type</span>
            </li>
          </ul>
        </div>

        <div className="intro-section">
          <div className="intro-section-title">
            <Maximize2 size={12} />
            Size
          </div>
          <p className="intro-text">Larger points = more downloads/likes</p>
        </div>

        <div className="intro-section">
          <div className="intro-section-title">
            <Move3D size={12} />
            Controls
          </div>
          <ul className="intro-list compact inline">
            <li><strong>Drag</strong> rotate</li>
            <li><strong>Scroll</strong> zoom</li>
            <li><strong>Click</strong> select</li>
          </ul>
        </div>

        <div className="intro-section">
          <div className="intro-section-title">
            <Search size={12} />
            Search
          </div>
          <ul className="intro-list compact">
            <li><kbd>⌘K</kbd> open search</li>
            <li><Sparkles size={10} className="intro-inline-icon" /> Fuzzy: <code>lama</code> finds llama</li>
          </ul>
        </div>
      </div>

      <div className="intro-footer">
        <label className="intro-checkbox">
          <input 
            type="checkbox" 
            checked={dontShowAgain}
            onChange={(e) => setDontShowAgain(e.target.checked)}
          />
          <span>Don't show again</span>
        </label>
        <button className="intro-dismiss" onClick={handleClose}>
          Got it
        </button>
      </div>
    </div>
  );
}

