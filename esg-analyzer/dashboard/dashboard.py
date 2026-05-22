"""
Streamlit Dashboard for ESG Analysis.
Interactive dashboard for visualizing ESG scores and metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from typing import Dict, List

from config import Config
from main import ESGAnalyzer

# Page configuration
st.set_page_config(
    page_title="ESG Analyzer Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = {}
if "summary" not in st.session_state:
    st.session_state.summary = {}


def load_results_from_disk() -> Dict:
    """Load all results from output directory."""
    config = Config()
    results = {}
    
    if not config.OUTPUTS_DIR.exists():
        return results
    
    for json_file in config.OUTPUTS_DIR.glob("*.json"):
        if json_file.name != "summary.json":
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    company_name = data.get("company_name", json_file.stem)
                    results[company_name] = data
            except Exception as e:
                st.warning(f"Error loading {json_file.name}: {str(e)}")
    
    return results


def load_summary_from_disk() -> Dict:
    """Load summary report from disk."""
    config = Config()
    summary_file = config.OUTPUTS_DIR / "summary.json"
    
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading summary: {str(e)}")
    
    return {}


def main():
    """Main dashboard application."""
    st.title("🌱 ESG Analyzer Dashboard")
    st.markdown("Comprehensive ESG analysis and visualization platform")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        page = st.radio(
            "Select View",
            ["Dashboard", "Company Comparison", "Time Series Analysis", "Process Reports"]
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.session_state.results = load_results_from_disk()
            st.session_state.summary = load_summary_from_disk()
            st.success("Data refreshed!")
    
    # Load data
    if not st.session_state.results:
        st.session_state.results = load_results_from_disk()
        st.session_state.summary = load_summary_from_disk()
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Company Comparison":
        show_comparison()
    elif page == "Time Series Analysis":
        show_time_series()
    elif page == "Process Reports":
        show_process_reports()


def show_dashboard():
    """Display main dashboard."""
    if not st.session_state.summary:
        st.info("No analysis results available. Process reports first.")
        return
    
    summary = st.session_state.summary
    
    # Overview metrics
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Companies Analyzed",
            summary.get("total_companies", 0)
        )
    
    with col2:
        overall_mean = summary.get("overall_scores", {}).get("mean", 0)
        st.metric(
            "Average ESG Score",
            f"{overall_mean:.1f}",
            delta=f"out of 100"
        )
    
    with col3:
        env_mean = summary.get("environmental_scores", {}).get("mean", 0)
        st.metric(
            "Avg Environmental",
            f"{env_mean:.1f}"
        )
    
    with col4:
        gov_mean = summary.get("governance_scores", {}).get("mean", 0)
        st.metric(
            "Avg Governance",
            f"{gov_mean:.1f}"
        )
    
    # Rankings
    st.header("ESG Score Rankings")
    rankings = summary.get("company_rankings", [])
    
    if rankings:
        df_rankings = pd.DataFrame(rankings, columns=["Company", "Score"])
        
        # Color code by score
        def score_color(score):
            if score >= 80:
                return "🟢"
            elif score >= 60:
                return "🟡"
            else:
                return "🔴"
        
        df_rankings["Status"] = df_rankings["Score"].apply(score_color)
        
        st.dataframe(
            df_rankings,
            use_container_width=True,
            hide_index=True
        )
        
        # Score distribution chart
        fig = px.bar(
            df_rankings,
            x="Company",
            y="Score",
            title="ESG Scores by Company",
            color="Score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category comparison
    st.header("Category Scores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        env_scores = summary.get("environmental_scores", {})
        st.write("**Environmental Scores**")
        st.json(env_scores)
    
    with col2:
        social_scores = summary.get("social_scores", {})
        st.write("**Social Scores**")
        st.json(social_scores)
    
    with col3:
        gov_scores = summary.get("governance_scores", {})
        st.write("**Governance Scores**")
        st.json(gov_scores)
    
    # Greenwashing analysis
    st.header("Greenwashing Risk Assessment")
    greenwashing_risks = summary.get("greenwashing_risks", {})
    
    if greenwashing_risks:
        risk_data = pd.DataFrame(
            [{"Company": k, "Risk": v} for k, v in greenwashing_risks.items()]
        )
        
        risk_colors = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red"
        }
        
        fig = px.bar(
            risk_data,
            x="Company",
            y="Risk",
            title="Greenwashing Risk Levels",
            color="Risk",
            color_discrete_map=risk_colors
        )
        st.plotly_chart(fig, use_container_width=True)


def show_comparison():
    """Display company comparison view."""
    st.header("Company Comparison")
    
    if not st.session_state.results:
        st.info("No results available")
        return
    
    companies = list(st.session_state.results.keys())
    selected = st.multiselect("Select companies to compare", companies)
    
    if selected:
        comparison_data = []
        
        for company in selected:
            result = st.session_state.results[company]
            comparison_data.append({
                "Company": company,
                "Overall Score": result.get("score", {}).get("overall_score", 0),
                "Environmental": result.get("score", {}).get("environmental_score", 0),
                "Social": result.get("score", {}).get("social_score", 0),
                "Governance": result.get("score", {}).get("governance_score", 0),
                "Greenwashing Risk": result.get("greenwashing", {}).get("greenwashing_risk", "N/A")
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Radar chart comparison
        categories = ["Environmental", "Social", "Governance"]
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            values = [
                row["Environmental"],
                row["Social"],
                row["Governance"]
            ]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=row["Company"]
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="ESG Category Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_time_series():
    """Display time series analysis."""
    st.header("Time Series Analysis")
    st.info("Time series visualization requires multiple years of data for the same company")
    
    # This would show ESG score trends over time if data is available
    st.write("Upload reports for different years to see trends")


def show_process_reports():
    """Display report processing interface."""
    st.header("Process ESG Reports")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF report",
        type=["pdf"],
        help="Select a sustainability/ESG report in PDF format"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name (optional)")
    
    with col2:
        report_year = st.number_input("Report Year (optional)", min_value=2000, max_value=2100)
    
    if uploaded_file and st.button("Analyze Report"):
        # Save uploaded file temporarily
        config = Config()
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        file_path = config.REPORTS_DIR / uploaded_file.name
        
        with st.spinner("Processing report..."):
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Analyze
                analyzer = ESGAnalyzer()
                result = analyzer.analyze_report(
                    file_path,
                    company_name=company_name if company_name else None,
                    report_year=int(report_year) if report_year else None
                )
                
                # Save result
                analyzer.save_results({result.company_name: result})
                
                st.success(f"Report analyzed successfully!")
                st.json(result.dict(exclude={"raw_text"}))
                
                # Refresh data
                st.session_state.results = load_results_from_disk()
                st.session_state.summary = load_summary_from_disk()
                
            except Exception as e:
                st.error(f"Error processing report: {str(e)}")


if __name__ == "__main__":
    main()
