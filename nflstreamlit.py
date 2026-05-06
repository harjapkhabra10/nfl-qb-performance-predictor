import streamlit as st
import pandas as pd

from modelpipeline import run_full_pipeline

st.set_page_config(
    page_title="QB Projections (2025 Weeks 13–18)",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_predictions() -> pd.DataFrame:
    """Run the full pipeline once and cache the result."""
    return run_full_pipeline()


st.title("🏈 QB Passing Projections — 2025 Weeks 13–18")
st.caption("Predicting attempts, completions, yards, comp%, and YPA for starting QBs.")

with st.spinner("Building dataset, training models, and generating predictions..."):
    future_results = load_predictions()

# clean the display a bit
display_cols = [
    "week",
    "team",
    "opponent_team",
    "player_name",
    "is_home",
    "predicted_attempts",
    "predicted_completions",
    "predicted_passing_yards",
    "predicted_comp_pct",
    "predicted_ypa",
]

future_results = future_results[display_cols].copy()
future_results["home/away"] = future_results["is_home"].map({1: "Home", 0: "Away"})
future_results = future_results.drop(columns=["is_home"])

# sidebar controls
st.sidebar.header("Filters")

view_mode = st.sidebar.radio(
    "View by:",
    ["Week", "Quarterback"],
    index=0,
)

if view_mode == "Week":
    available_weeks = sorted(future_results["week"].unique())
    selected_week = st.sidebar.selectbox("Select week", available_weeks)

    week_df = future_results[future_results["week"] == selected_week].copy()
    week_df = week_df.sort_values(
        "predicted_passing_yards", ascending=False
    ).reset_index(drop=True)

    st.subheader(f"Week {selected_week} — All QBs")
    st.dataframe(
        week_df[
            [
                "team",
                "opponent_team",
                "player_name",
                "home/away",
                "predicted_attempts",
                "predicted_completions",
                "predicted_passing_yards",
                "predicted_comp_pct",
                "predicted_ypa",
            ]
        ],
        use_container_width=True,
    )

elif view_mode == "Quarterback":
    qbs = sorted(future_results["player_name"].dropna().unique())
    selected_qb = st.sidebar.selectbox("Select QB", qbs)

    qb_df = future_results[future_results["player_name"] == selected_qb].copy()
    qb_df = qb_df.sort_values("week").reset_index(drop=True)

    st.subheader(f"{selected_qb} — Weeks 13–18 Projections")

    st.dataframe(
        qb_df[
            [
                "week",
                "team",
                "opponent_team",
                "home/away",
                "predicted_attempts",
                "predicted_completions",
                "predicted_passing_yards",
                "predicted_comp_pct",
                "predicted_ypa",
            ]
        ],
        use_container_width=True,
    )