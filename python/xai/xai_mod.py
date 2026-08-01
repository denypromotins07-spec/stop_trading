"""
XAI Module Root - Pushes explainability insights to SOUL.md for model introspection.
Integrates SHAP explanations and drift attribution to help the bot understand
why specific trading decisions were made, especially mistakes.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np

from .shap_explainer import get_explainer_actor, SHAPExplainerActor
from .drift_attribution import get_drift_engine, DriftAttributionEngine, DriftEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XAIModule:
    """
    Central module for Explainable AI insights.
    Coordinates SHAP explanations and drift attribution, pushing results to SOUL.md.
    """
    
    def __init__(self, 
                 soul_md_path: str = "SOUL.md",
                 feature_names: Optional[List[str]] = None,
                 max_history: int = 1000):
        """
        Initialize XAI module.
        
        Args:
            soul_md_path: Path to SOUL.md file for logging insights
            feature_names: List of feature names for interpretability
            max_history: Maximum number of explanations to keep in memory
        """
        self.soul_md_path = Path(soul_md_path)
        self.feature_names = feature_names or []
        self.max_history = max_history
        
        self._explainer_actor: Optional[SHAPExplainerActor] = None
        self._drift_engine: Optional[DriftAttributionEngine] = None
        
        self._explanation_history: List[Dict[str, Any]] = []
        self._mistake_log: List[Dict[str, Any]] = []
        
        self._is_initialized = False
    
    async def initialize(self, model: Any = None, 
                         background_data: Optional[np.ndarray] = None) -> bool:
        """
        Initialize XAI components.
        
        Args:
            model: Trained model for SHAP explanation
            background_data: Background dataset for SHAP
            
        Returns:
            True if initialization successful
        """
        try:
            # Initialize SHAP explainer actor
            if model is not None:
                self._explainer_actor = get_explainer_actor()
                if self._explainer_actor is not None and background_data is not None:
                    await self._explainer_actor.set_model.remote(
                        model, self.feature_names
                    )
                    await self._explainer_actor.set_background_data.remote(background_data)
                    logger.info("SHAP explainer initialized")
            
            # Initialize drift engine
            if self.feature_names:
                self._drift_engine = get_drift_engine(self.feature_names)
                logger.info("Drift attribution engine initialized")
            
            self._is_initialized = True
            logger.info("XAI module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize XAI module: {e}")
            return False
    
    async def explain_prediction(self, features: np.ndarray, 
                                  prediction: float,
                                  actual_outcome: Optional[float] = None,
                                  trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction/trade.
        
        Args:
            features: Feature vector for the prediction
            prediction: Model prediction value
            actual_outcome: Actual outcome (for mistake analysis)
            trade_id: Optional trade identifier
            
        Returns:
            Explanation dictionary
        """
        if not self._is_initialized:
            return {"error": "XAI module not initialized"}
        
        explanation = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade_id,
            "prediction": float(prediction),
            "features": features.tolist() if hasattr(features, 'tolist') else list(features),
            "feature_names": self.feature_names
        }
        
        # Get SHAP explanation if available
        if self._explainer_actor is not None:
            try:
                feat_array = np.array(features).reshape(1, -1)
                shap_result = await self._explainer_actor.compute_shap_values.remote(feat_array)
                
                if "error" not in shap_result and shap_result.get("shap_values") is not None:
                    explanation["shap_values"] = shap_result["shap_values"][0].tolist()
                    explanation["base_value"] = float(shap_result["base_value"])
                    
                    # Calculate top contributing features
                    shap_vals = shap_result["shap_values"][0]
                    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
                    explanation["top_features"] = [
                        {
                            "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                            "contribution": float(shap_vals[i]),
                            "direction": "positive" if shap_vals[i] > 0 else "negative"
                        }
                        for i in sorted_indices
                    ]
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                explanation["shap_error"] = str(e)
        
        # Update drift engine
        if self._drift_engine is not None:
            drift_result = self._drift_engine.update(
                features.reshape(1, -1) if len(features.shape) == 1 else features,
                np.array([prediction])
            )
            if drift_result:
                explanation["drift_detected"] = True
                explanation["drift_psi"] = drift_result.psi_score
                explanation["drift_ks_pvalue"] = drift_result.ks_p_value
        
        # Check if this was a mistake (if actual outcome provided)
        if actual_outcome is not None:
            error = abs(prediction - actual_outcome)
            explanation["actual_outcome"] = float(actual_outcome)
            explanation["error"] = error
            explanation["was_mistake"] = error > 0.1  # Configurable threshold
            
            if explanation["was_mistake"]:
                self._log_mistake(explanation)
        
        # Store in history
        self._explanation_history.append(explanation)
        if len(self._explanation_history) > self.max_history:
            self._explanation_history.pop(0)
        
        return explanation
    
    def _log_mistake(self, explanation: Dict[str, Any]):
        """Log a trading mistake with full explanation for SOUL.md."""
        mistake_entry = {
            "timestamp": explanation["timestamp"],
            "trade_id": explanation.get("trade_id", "unknown"),
            "prediction": explanation["prediction"],
            "actual": explanation.get("actual_outcome"),
            "error": explanation.get("error"),
            "top_contributing_factors": explanation.get("top_features", []),
            "drift_context": {
                "detected": explanation.get("drift_detected", False),
                "psi": explanation.get("drift_psi"),
            } if explanation.get("drift_detected") else None
        }
        
        self._mistake_log.append(mistake_entry)
        if len(self._mistake_log) > 100:
            self._mistake_log.pop(0)
        
        logger.warning(
            f"Mistake logged: trade={mistake_entry['trade_id']}, "
            f"error={mistake_entry['error']:.4f}, "
            f"top_factor={mistake_entry['top_contributing_factors'][0]['name'] if mistake_entry['top_contributing_factors'] else 'unknown'}"
        )
    
    async def write_to_soul_md(self):
        """
        Write XAI insights to SOUL.md file.
        Includes recent explanations, mistake analysis, and drift summary.
        """
        if not self._is_initialized:
            return
        
        try:
            content = []
            content.append("# XAI Insights - Model Introspection\n")
            content.append(f"*Last updated: {datetime.utcnow().isoformat()}*\n")
            
            # Section 1: Recent Mistakes Analysis
            content.append("## Recent Trading Mistakes\n")
            if self._mistake_log:
                content.append(f"Total mistakes tracked: {len(self._mistake_log)}\n\n")
                
                # Aggregate mistake patterns
                feature_mistake_counts = {}
                for mistake in self._mistake_log[-20:]:  # Last 20 mistakes
                    for factor in mistake.get("top_contributing_factors", [])[:3]:
                        fname = factor["name"]
                        feature_mistake_counts[fname] = feature_mistake_counts.get(fname, 0) + 1
                
                if feature_mistake_counts:
                    content.append("### Features Most Associated with Mistakes:\n")
                    sorted_features = sorted(feature_mistake_counts.items(), key=lambda x: x[1], reverse=True)
                    for feat, count in sorted_features[:10]:
                        content.append(f"- **{feat}**: involved in {count} mistakes\n")
                    content.append("\n")
                
                # Detailed recent mistakes
                content.append("### Recent Mistake Details:\n")
                for i, mistake in enumerate(reversed(self._mistake_log[-5:])):
                    content.append(f"\n**Mistake #{len(self._mistake_log) - i}** (Trade: {mistake['trade_id']})\n")
                    content.append(f"- Prediction: {mistake['prediction']:.4f}\n")
                    content.append(f"- Actual: {mistake['actual']:.4f}\n")
                    content.append(f"- Error: {mistake['error']:.4f}\n")
                    
                    if mistake.get("top_contributing_factors"):
                        content.append("- Top contributing factors:\n")
                        for factor in mistake["top_contributing_factors"][:3]:
                            content.append(f"  - {factor['name']}: {factor['contribution']:.4f} ({factor['direction']})\n")
                    
                    if mistake.get("drift_context") and mistake["drift_context"]["detected"]:
                        content.append(f"- ⚠️ Drift detected (PSI: {mistake['drift_context']['psi']:.4f})\n")
            else:
                content.append("*No mistakes recorded yet.*\n")
            
            content.append("\n---\n")
            
            # Section 2: Drift Summary
            content.append("## Feature Drift Status\n")
            if self._drift_engine:
                drift_summary = self._drift_engine.get_drift_summary()
                if drift_summary.get("status") == "monitoring":
                    latest = drift_summary.get("latest_drift", {})
                    content.append(f"- Current PSI Score: {latest.get('psi_score', 'N/A')}\n")
                    content.append(f"- KS Test P-value: {latest.get('ks_p_value', 'N/A')}\n")
                    content.append(f"- Mean Prediction Shift: {latest.get('mean_shift', 'N/A')}\n")
                    
                    severity_counts = drift_summary.get("severity_counts", {})
                    if any(severity_counts.values()):
                        content.append("\n### Drift Events by Severity:\n")
                        for severity, count in severity_counts.items():
                            if count > 0:
                                content.append(f"- {severity.capitalize()}: {count}\n")
                    
                    top_drifting = drift_summary.get("top_drifting_features", [])
                    if top_drifting:
                        content.append("\n### Most Unstable Features:\n")
                        for feat, count in top_drifting[:5]:
                            content.append(f"- {feat}: {count} drift events\n")
                else:
                    content.append("*Drift monitoring not yet calibrated.*\n")
            else:
                content.append("*Drift engine not initialized.*\n")
            
            content.append("\n---\n")
            
            # Section 3: Feature Importance Summary
            content.append("## Global Feature Importance\n")
            if self._explainer_actor is not None:
                try:
                    importance = await self._explainer_actor.get_feature_importance_summary.remote(15)
                    if importance:
                        content.append("| Feature | Mean |SHAP| |\n")
                        content.append("|---------|----------|\n")
                        for feat, imp in importance.items():
                            content.append(f"| {feat} | {imp:.6f} |\n")
                    else:
                        content.append("*No SHAP data available yet.*\n")
                except Exception as e:
                    content.append(f"*Error retrieving feature importance: {e}*\n")
            else:
                content.append("*SHAP explainer not initialized.*\n")
            
            content.append("\n---\n")
            content.append("\n*This file is auto-generated by the XAI module for model introspection.*\n")
            
            # Write to file
            with open(self.soul_md_path, 'w') as f:
                f.write("".join(content))
            
            logger.info(f"XAI insights written to {self.soul_md_path}")
            
        except Exception as e:
            logger.error(f"Failed to write to SOUL.md: {e}")
    
    def get_mistake_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in trading mistakes."""
        if not self._mistake_log:
            return {"status": "no_data"}
        
        # Analyze patterns
        avg_error = np.mean([m["error"] for m in self._mistake_log])
        max_error = max(m["error"] for m in self._mistake_log)
        
        # Feature involvement
        feature_involvement = {}
        for mistake in self._mistake_log:
            for factor in mistake.get("top_contributing_factors", []):
                fname = factor["name"]
                if fname not in feature_involvement:
                    feature_involvement[fname] = {"count": 0, "total_impact": 0.0}
                feature_involvement[fname]["count"] += 1
                feature_involvement[fname]["total_impact"] += abs(factor["contribution"])
        
        # Calculate average impact per feature
        for fname in feature_involvement:
            count = feature_involvement[fname]["count"]
            feature_involvement[fname]["avg_impact"] = feature_involvement[fname]["total_impact"] / count
        
        # Drift correlation
        drift_correlated = sum(1 for m in self._mistake_log if m.get("drift_context", {}).get("detected", False))
        
        return {
            "total_mistakes": len(self._mistake_log),
            "average_error": avg_error,
            "max_error": max_error,
            "feature_involvement": feature_involvement,
            "drift_correlated_mistakes": drift_correlated,
            "drift_correlation_rate": drift_correlated / len(self._mistake_log) if self._mistake_log else 0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return module health status."""
        return {
            "initialized": self._is_initialized,
            "explainer_available": self._explainer_actor is not None,
            "drift_engine_available": self._drift_engine is not None,
            "explanations_cached": len(self._explanation_history),
            "mistakes_logged": len(self._mistake_log),
            "feature_count": len(self.feature_names)
        }


# Module singleton
_xai_module: Optional[XAIModule] = None


def get_xai_module(feature_names: Optional[List[str]] = None, 
                   soul_md_path: str = "SOUL.md") -> XAIModule:
    """Get or create the global XAI module."""
    global _xai_module
    
    if _xai_module is None:
        _xai_module = XAIModule(
            soul_md_path=soul_md_path,
            feature_names=feature_names
        )
        logger.info(f"Created XAI module with {len(feature_names or [])} features")
    
    return _xai_module


async def initialize_xai(model: Any = None, 
                         background_data: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None) -> bool:
    """Initialize the global XAI module."""
    module = get_xai_module(feature_names)
    return await module.initialize(model, background_data)


if __name__ == "__main__":
    # Test the XAI module
    import sys
    sys.path.insert(0, '/workspace/python')
    
    asyncio.run(asyncio.sleep(0))  # Ensure event loop exists
    
    # Create test module
    feature_names = ["spread", "volatility", "momentum", "order_imbalance", "micro_price"]
    module = XAIModule(feature_names=feature_names)
    
    # Mock model initialization (in real use, pass actual XGBoost model)
    asyncio.run(module.initialize())
    
    # Simulate some predictions
    for i in range(10):
        features = np.random.randn(len(feature_names))
        prediction = np.sum(features) * 0.1
        actual = prediction + np.random.randn() * 0.05
        
        explanation = asyncio.run(module.explain_prediction(
            features, prediction, actual, trade_id=f"trade_{i}"
        ))
        
        print(f"Trade {i}: prediction={prediction:.4f}, actual={actual:.4f}")
        if explanation.get("was_mistake"):
            print(f"  → Mistake! Error: {explanation['error']:.4f}")
            if explanation.get("top_features"):
                print(f"  → Top factor: {explanation['top_features'][0]}")
    
    # Write to SOUL.md
    asyncio.run(module.write_to_soul_md())
    
    print(f"\nMistake patterns: {module.get_mistake_patterns()}")
    print(f"Health: {module.health_check()}")
