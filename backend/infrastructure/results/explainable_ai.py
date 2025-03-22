from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from darts.explainability.explainer import (
    ComponentBasedExplainabilityResult,
    HorizonBasedExplainabilityResult,
)
from darts.explainability.shap_explainer import ShapExplainabilityResult
from darts.explainability.tft_explainer import TFTExplainabilityResult


class ExplainabilityHandler:
    @staticmethod
    def explain_model(
        model: Any,
        series: Any,
        prediction: Any,
    ) -> (
        ShapExplainabilityResult
        | TFTExplainabilityResult
        | ComponentBasedExplainabilityResult
        | HorizonBasedExplainabilityResult
    ):
        if hasattr(model, "explain"):
            return model.explain(series, prediction)
        else:
            raise ValueError("This model doesn't have built-in explainability.")

    @staticmethod
    def visualize_shap(result: ShapExplainabilityResult):
        st.subheader("SHAP Explainability Results")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=result.shap_values.mean(axis=0), y=result.feature_names, ax=ax)
        ax.set_title("Feature Importance (SHAP values)")
        st.pyplot(fig)

    @staticmethod
    def visualize_tft(result: TFTExplainabilityResult):
        st.subheader("TFT Explainability Results")

        st.write("Static Variables Importance:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=result.static_variables_importance.values(), y=result.static_variables_importance.keys(), ax=ax)
        ax.set_title("Static Variables Importance")
        st.pyplot(fig)

        st.write("Temporal Variables Importance:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=result.temporal_variables_importance.values(),
            y=result.temporal_variables_importance.keys(),
            ax=ax,
        )
        ax.set_title("Temporal Variables Importance")
        st.pyplot(fig)

    @staticmethod
    def visualize_component_based(result: ComponentBasedExplainabilityResult):
        st.subheader("Component-Based Explainability Results")
        for component, values in result.components.items():
            st.write(f"{component} Component:")
            st.line_chart(values)

    @staticmethod
    def visualize_horizon_based(result: HorizonBasedExplainabilityResult):
        st.subheader("Horizon-Based Explainability Results")
        for horizon, importance in result.importance_by_horizon.items():
            st.write(f"Horizon {horizon}:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance.values(), y=importance.keys(), ax=ax)
            ax.set_title(f"Feature Importance for Horizon {horizon}")
            st.pyplot(fig)

    @staticmethod
    def explain_and_visualize(model: Any, series: Any, prediction: Any):
        try:
            result = ExplainabilityHandler.explain_model(model, series, prediction)

            if isinstance(result, ShapExplainabilityResult):
                ExplainabilityHandler.visualize_shap(result)
            elif isinstance(result, TFTExplainabilityResult):
                ExplainabilityHandler.visualize_tft(result)
            elif isinstance(result, ComponentBasedExplainabilityResult):
                ExplainabilityHandler.visualize_component_based(result)
            elif isinstance(result, HorizonBasedExplainabilityResult):
                ExplainabilityHandler.visualize_horizon_based(result)
            else:
                st.warning("Unknown explainability result type.")
        except ValueError as e:
            st.warning(str(e))
