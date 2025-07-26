#!/usr/bin/env python3
"""
Complete Streamlit Version of Iota Calculator
Compatible with your existing sim.py and requirements.txt
Run with: streamlit run streamlit_iota_calculator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Any, Optional
import warnings
import re
import io
import contextlib

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Iota Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import dependencies with error handling
@st.cache_data
def check_dependencies():
    """Check if all dependencies are available."""
    missing = []
    
    try:
        from scipy import stats
    except ImportError:
        missing.append("scipy")
    
    try:
        from sim import fetch_backtest, calculate_portfolio_returns
    except ImportError:
        missing.append("sim.py")
    
    return missing

# Check dependencies at startup
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing dependencies: {', '.join(missing_deps)}")
    st.markdown("""
    **Required files:**
    - `sim.py` - Your portfolio calculation module
    - Install scipy: `pip install scipy`
    
    Make sure `sim.py` is in the same directory as this Streamlit app.
    """)
    st.stop()

# Now import everything
from scipy import stats
from sim import fetch_backtest, calculate_portfolio_returns

# ===== CORE FUNCTIONS (From your original Iota.py) =====

def parse_exclusion_input(user_str: str) -> List[Tuple[date, date]]:
    """Return list of date ranges from user string."""
    if not user_str.strip():
        return []
    out: List[Tuple[date, date]] = []
    for token in user_str.split(","):
        token = token.strip()
        if not token:
            continue
        found = re.findall(r"\d{4}-\d{2}-\d{2}", token)
        if len(found) != 2:
            st.warning(f"Skipping unparsable exclusion token: '{token}'.")
            continue
        a, b = [datetime.strptime(d, "%Y-%m-%d").date() for d in found]
        out.append((min(a, b), max(a, b)))
    return out

def cumulative_return(daily_pct: pd.Series) -> float:
    """Total compounded return over the period (decimal)."""
    daily_dec = daily_pct.dropna() / 100.0
    return float(np.prod(1 + daily_dec) - 1) if not daily_dec.empty else 0.0

def window_cagr(daily_pct: pd.Series) -> float:
    """Compounded annual growth rate over window."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    total_return = np.prod(1 + daily_dec) - 1
    days = len(daily_dec)
    if days < 2:
        return 0.0
    try:
        cagr = (1 + total_return) ** (252 / days) - 1
        return cagr
    except (FloatingPointError, ValueError):
        return 0.0

def sharpe_ratio(daily_pct: pd.Series) -> float:
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.std(ddof=0) == 0:
        return 0.0
    return (daily_dec.mean() / daily_dec.std(ddof=0)) * np.sqrt(252)

def sortino_ratio(daily_pct: pd.Series) -> float:
    """Enhanced Sortino ratio with proper zero-downside handling."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    
    downside = daily_dec[daily_dec < 0]
    mean_return = daily_dec.mean()
    
    if len(downside) == 0:
        if mean_return > 0:
            return np.inf
        else:
            return 0.0
    
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return 0.0
    
    return (mean_return / downside_std) * np.sqrt(252)

def assess_sample_reliability(n_is: int, n_oos: int) -> str:
    """Assess statistical reliability based on sample sizes."""
    min_size = min(n_is, n_oos)
    
    if min_size >= 378:
        return "HIGH_CONFIDENCE"
    elif min_size >= 189:
        return "MODERATE_CONFIDENCE"  
    elif min_size >= 90:
        return "LOW_CONFIDENCE"
    else:
        return "INSUFFICIENT_DATA"

def build_slices(is_ret: pd.Series, slice_len: int, n_slices: int, overlap: bool) -> List[pd.Series]:
    """Return list of IS slices each of length slice_len."""
    total_is = len(is_ret)
    max_start = total_is - slice_len

    if max_start < 0:
        return []

    if not overlap:
        slices: List[pd.Series] = []
        end_idx = total_is
        while len(slices) < n_slices and end_idx >= slice_len:
            seg = is_ret.iloc[end_idx - slice_len : end_idx]
            if len(seg) == slice_len:
                slices.append(seg)
            end_idx -= slice_len
        return slices

    if n_slices == 1:
        starts = [max_start]
    else:
        starts = np.linspace(0, max_start, n_slices, dtype=int).tolist()
    starts = sorted(dict.fromkeys(starts))

    return [is_ret.iloc[s : s + slice_len] for s in starts]

def compute_iota(is_metric: float, oos_metric: float, n_oos: int, n_ref: int = 252, eps: float = 1e-6, 
                 lower_is_better: bool = False, is_values: np.ndarray = None) -> float:
    """INTUITIVE standardized iota calculation."""
    if np.isinf(oos_metric):
        return 2.0 if not lower_is_better else -2.0
    
    if is_values is None:
        return 0.0
    
    finite_is = is_values[np.isfinite(is_values)]
    if len(finite_is) < 2:
        return 0.0
    
    is_median = np.median(finite_is)
    is_std = np.std(finite_is, ddof=1)
    
    if is_std < eps:
        return 0.0
    
    standardized_diff = (oos_metric - is_median) / is_std
    
    if lower_is_better:
        standardized_diff = -standardized_diff
    
    w = min(1.0, np.sqrt(n_oos / n_ref))
    
    return w * standardized_diff

def iota_to_persistence_rating(iota_val: float, max_rating: int = 500) -> int:
    """Convert iota to persistence rating."""
    if not np.isfinite(iota_val):
        return 100
    
    k = 0.5
    rating = 100 * np.exp(k * iota_val)
    return max(1, min(max_rating, int(round(rating))))

def interpret_iota_directly(iota_val: float) -> str:
    """Direct interpretation of standardized iota values."""
    if not np.isfinite(iota_val):
        return "UNDEFINED"
    
    if iota_val >= 2.0:
        return "🔥 EXCEPTIONAL: OOS >2σ above IS median"
    elif iota_val >= 1.0:
        return "✅ EXCELLENT: OOS >1σ above IS median"
    elif iota_val >= 0.5:
        return "👍 GOOD: OOS >0.5σ above IS median"
    elif iota_val >= 0.1:
        return "📈 SLIGHT_IMPROVEMENT: OOS mildly above IS median"
    elif iota_val >= -0.1:
        return "➡️ NEUTRAL: OOS ≈ IS median"
    elif iota_val >= -0.5:
        return "⚠️ CAUTION: OOS below IS median"
    elif iota_val >= -1.0:
        return "🚨 WARNING: OOS >0.5σ below IS median"
    elif iota_val >= -2.0:
        return "🔴 ALERT: OOS >1σ below IS median"
    else:
        return "💀 CRITICAL: OOS >2σ below IS median"

def standard_bootstrap_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                                 n_bootstrap: int, confidence_level: float, 
                                 lower_is_better: bool) -> Tuple[float, float]:
    """Standard bootstrap for non-overlapping slices."""
    try:
        bootstrap_iotas = []
        for _ in range(n_bootstrap):
            try:
                boot_sample = np.random.choice(is_values, size=len(is_values), replace=True)
                boot_median = np.median(boot_sample)
                boot_iota = compute_iota(boot_median, oos_value, n_oos, lower_is_better=lower_is_better, 
                                       is_values=boot_sample)
                if np.isfinite(boot_iota):
                    bootstrap_iotas.append(boot_iota)
            except Exception:
                continue
        
        if len(bootstrap_iotas) < 50:
            return np.nan, np.nan
        
        alpha = 1 - confidence_level
        return tuple(np.percentile(bootstrap_iotas, [100 * alpha/2, 100 * (1 - alpha/2)]))
    except Exception:
        return np.nan, np.nan

def bootstrap_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int, 
                             n_bootstrap: int = 1000, confidence_level: float = 0.95,
                             lower_is_better: bool = False, overlap: bool = True) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(is_values) < 3:
        return np.nan, np.nan
    
    return standard_bootstrap_confidence(is_values, oos_value, n_oos, n_bootstrap, 
                                       confidence_level, lower_is_better)

def wilcoxon_iota_test(is_values: np.ndarray, oos_value: float, n_oos: int,
                      lower_is_better: bool = False) -> Tuple[float, bool]:
    """Simple Wilcoxon test."""
    if len(is_values) < 6:
        return np.nan, False
    
    slice_iotas = []
    for is_val in is_values:
        iota_val = compute_iota(is_val, oos_value, n_oos, lower_is_better=lower_is_better, 
                               is_values=is_values)
        if np.isfinite(iota_val):
            slice_iotas.append(iota_val)
    
    if len(slice_iotas) < 6:
        return np.nan, False
    
    try:
        _, p_value = stats.wilcoxon(slice_iotas, alternative='two-sided')
        return p_value, p_value < 0.05
    except (ValueError, ZeroDivisionError):
        return np.nan, False

def compute_iota_with_stats(is_values: np.ndarray, oos_value: float, n_oos: int, 
                           metric_name: str = "metric", lower_is_better: bool = False,
                           overlap: bool = True) -> Dict[str, Any]:
    """Enhanced iota computation with statistical tests."""
    if len(is_values) == 0:
        return {
            'iota': np.nan,
            'persistence_rating': 100,
            'confidence_interval': (np.nan, np.nan),
            'p_value': np.nan,
            'significant': False,
            'median_is': np.nan,
            'iqr_is': (np.nan, np.nan)
        }
    
    median_is = np.median(is_values)
    q25_is, q75_is = np.percentile(is_values, [25, 75])
    
    iota = compute_iota(median_is, oos_value, n_oos, lower_is_better=lower_is_better, is_values=is_values)
    persistence_rating = iota_to_persistence_rating(iota)
    
    ci_lower, ci_upper = bootstrap_iota_confidence(is_values, oos_value, n_oos, 
                                                  lower_is_better=lower_is_better, overlap=overlap)
    
    p_value, significant = wilcoxon_iota_test(is_values, oos_value, n_oos, 
                                            lower_is_better=lower_is_better)
    
    return {
        'iota': iota,
        'persistence_rating': persistence_rating,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value,
        'significant': significant,
        'median_is': median_is,
        'iqr_is': (q25_is, q75_is)
    }

def format_sortino_output(sortino_val: float) -> str:
    """Special formatting for Sortino ratio including infinite values."""
    if np.isinf(sortino_val):
        return "∞ (no downside)"
    elif np.isnan(sortino_val):
        return "NaN"
    else:
        return f"{sortino_val:.3f}"

# ===== STREAMLIT APP =====

def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .success-card {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
        }
        .warning-card {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📊 Iota Calculator</h1>', unsafe_allow_html=True)
    st.markdown("**Quantify your strategy's out-of-sample performance vs. historical expectations**")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📊 Results", "📚 Help"])
    
    # Configuration Tab
    with tab1:
        st.header("Analysis Configuration")
        
        # Main configuration form
        with st.form("analysis_form"):
            st.subheader("📝 Required Information")
            
            # Symphony URL
            url = st.text_input(
                "Composer Symphony URL *",
                help="Enter the full URL of your Composer symphony",
                placeholder="https://app.composer.trade/symphony/..."
            )
            
            # Date configuration
            col1, col2 = st.columns(2)
            with col1:
                early_date = st.date_input(
                    "Data Start Date:",
                    value=date(2000, 1, 1),
                    help="How far back to fetch historical data"
                )
            
            with col2:
                today_date = st.date_input(
                    "Data End Date:",
                    value=date.today(),
                    help="End date for data fetching"
                )
            
            # OOS start date - this is crucial
            oos_start = st.date_input(
                "Out-of-Sample Start Date *",
                value=date.today() - timedelta(days=730),  # Default 2 years ago
                help="⚠️ CRITICAL: Date when your 'live trading' or out-of-sample period begins. Everything before this is historical backtest data, everything after is 'real world' performance."
            )
            
            st.markdown("---")
            st.subheader("⚙️ Analysis Parameters")
            
            # Analysis parameters in columns
            col1, col2 = st.columns(2)
            with col1:
                n_slices = st.number_input(
                    "Number of IS Slices:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    help="How many historical periods to compare against (more = better statistics, slower analysis)"
                )
            
            with col2:
                overlap = st.checkbox(
                    "Allow Overlapping Slices",
                    value=True,
                    help="Whether historical comparison periods can overlap (recommended: True for more data)"
                )
            
            # Optional exclusion windows
            st.subheader("🚫 Exclusion Windows (Optional)")
            exclusions_str = st.text_area(
                "Exclude specific date ranges:",
                help="Exclude market crashes, unusual periods, etc. Format: YYYY-MM-DD to YYYY-MM-DD, separated by commas",
                placeholder="2020-03-01 to 2020-05-01, 2022-01-01 to 2022-02-01"
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Run Iota Analysis", type="primary")
            
            if submitted:
                # Validate inputs
                if not url.strip():
                    st.error("❌ Please enter a Composer Symphony URL")
                elif oos_start <= early_date:
                    st.error("❌ Out-of-Sample start date must be after the data start date")
                elif oos_start >= today_date:
                    st.error("❌ Out-of-Sample start date must be before the data end date")
                else:
                    # Store in session state to pass to results tab
                    st.session_state.analysis_config = {
                        'url': url,
                        'early_date': early_date,
                        'today_date': today_date,
                        'oos_start': oos_start,
                        'n_slices': n_slices,
                        'overlap': overlap,
                        'exclusions_str': exclusions_str
                    }
                    st.session_state.run_analysis = True
                    st.success("✅ Configuration saved! Go to the 'Results' tab to see your analysis.")
        
        # Show example configuration
        with st.expander("💡 Example Configuration"):
            st.markdown("""
            **Example Strategy Analysis:**
            - **Symphony URL**: `https://app.composer.trade/symphony/xyz123`
            - **Data Period**: 2015-01-01 to 2024-12-31
            - **OOS Start**: 2022-01-01 *(everything after this is "live performance")*
            - **IS Slices**: 100 *(creates 100 historical comparison periods)*
            - **Overlap**: Yes *(more data for statistics)*
            
            **What this means:**
            - Your backtest covers 2015-2021 (7 years of historical data)
            - Your "out-of-sample" period is 2022-2024 (3 years of live performance)
            - The tool creates 100 different 3-year periods from 2015-2021 to compare against your 2022-2024 performance
            """)

    # Results Tab
    with tab2:
        st.header("Analysis Results")
        
        if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
            config = st.session_state.analysis_config
            
            try:
                # Parse exclusions
                exclusions = parse_exclusion_input(config['exclusions_str'])
                if exclusions:
                    st.info(f"📋 {len(exclusions)} exclusion window(s) will be applied")
                
                # Run the analysis with progress tracking
                with st.spinner("🔄 Fetching backtest data from Composer..."):
                    alloc, sym_name, tickers = fetch_backtest(
                        config['url'], 
                        config['early_date'].strftime("%Y-%m-%d"), 
                        config['today_date'].strftime("%Y-%m-%d")
                    )
                
                st.success(f"✅ Successfully fetched data for strategy: **{sym_name}**")
                
                with st.spinner("🧮 Calculating portfolio returns..."):
                    # Capture stdout during portfolio calculation
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
                
                # Convert index to date
                daily_ret.index = pd.to_datetime(daily_ret.index).date
                
                st.info(f"📈 Loaded {len(daily_ret)} days of return data")
                
                # Apply exclusions
                if exclusions:
                    mask = pd.Series(True, index=daily_ret.index)
                    for s, e in exclusions:
                        mask &= ~((daily_ret.index >= s) & (daily_ret.index <= e))
                    removed = (~mask).sum()
                    daily_ret = daily_ret[mask]
                    st.warning(f"🚫 Excluded {removed} days across {len(exclusions)} window(s)")
                
                # Split data
                oos_start_dt = config['oos_start']
                is_ret = daily_ret[daily_ret.index < oos_start_dt]
                oos_ret = daily_ret[daily_ret.index >= oos_start_dt]
                
                if len(is_ret) < 30 or len(oos_ret) < 30:
                    st.error("❌ Insufficient data: Need at least 30 days in both IS and OOS periods")
                    return
                
                n_oos = len(oos_ret)
                n_is = len(is_ret)
                
                # Show data split summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("In-Sample Days", f"{n_is}")
                    st.caption(f"{is_ret.index[0]} to {is_ret.index[-1]}")
                with col2:
                    st.metric("Out-of-Sample Days", f"{n_oos}")
                    st.caption(f"{oos_ret.index[0]} to {oos_ret.index[-1]}")
                with col3:
                    reliability = assess_sample_reliability(n_is, n_oos)
                    st.metric("Reliability", reliability.replace("_", " "))
                
                with st.spinner("📊 Running Iota analysis..."):
                    # Calculate OOS metrics
                    ar_oos = window_cagr(oos_ret)
                    sh_oos = sharpe_ratio(oos_ret)
                    cr_oos = cumulative_return(oos_ret)
                    so_oos = sortino_ratio(oos_ret)
                    
                    # Build IS slices
                    slice_len = n_oos
                    slices = build_slices(is_ret, slice_len, config['n_slices'], config['overlap'])
                    
                    if not slices:
                        st.error("❌ Could not create IS slices of required length")
                        return
                    
                    # Calculate IS metrics for each slice
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    rows = []
                    for i, s in enumerate(slices[::-1], 1):
                        progress_bar.progress(i / len(slices))
                        status_text.text(f"Processing slice {i}/{len(slices)}")
                        
                        rows.append({
                            "slice": i,
                            "start": s.index[0],
                            "end": s.index[-1],
                            "ar_is": window_cagr(s),
                            "sh_is": sharpe_ratio(s),
                            "cr_is": cumulative_return(s),
                            "so_is": sortino_ratio(s)
                        })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    df = pd.DataFrame(rows)
                    
                    # Compute iota statistics
                    ar_stats = compute_iota_with_stats(df["ar_is"].values, ar_oos, n_oos, "Annualized Return", overlap=config['overlap'])
                    sh_stats = compute_iota_with_stats(df["sh_is"].values, sh_oos, n_oos, "Sharpe Ratio", overlap=config['overlap'])
                    cr_stats = compute_iota_with_stats(df["cr_is"].values, cr_oos, n_oos, "Cumulative Return", overlap=config['overlap'])
                    so_stats = compute_iota_with_stats(df["so_is"].values, so_oos, n_oos, "Sortino Ratio", overlap=config['overlap'])
                
                # Display results
                display_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                               ar_oos, sh_oos, cr_oos, so_oos, reliability, config)
                
                # Reset the flag
                st.session_state.run_analysis = False
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Please configure and run your analysis in the 'Configuration' tab first.")
            
            # Show what the analysis will do
            st.markdown("""
            ### What will the analysis show you?
            
            The Iota Calculator will provide:
            
            1. **📊 Performance Metrics Analysis**
               - Sharpe Ratio, Cumulative Return, Sortino Ratio, Annualized Return
               - How your OOS performance compares to historical expectations
            
            2. **🎯 Iota Scores**
               - Standardized measure of performance deviation
               - Positive = Better than expected, Negative = Worse than expected
            
            3. **📈 Statistical Significance**
               - P-values and confidence intervals
               - How confident we can be in the results
            
            4. **📋 Persistence Ratings**
               - Easy-to-understand 0-500 scale
               - 100 = Neutral, >100 = Outperformance, <100 = Underperformance
            """)

    # Help Tab
    with tab3:
        show_help_content()

def display_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                   ar_oos, sh_oos, cr_oos, so_oos, reliability, config):
    """Display the analysis results."""
    
    st.success(f"🎉 Analysis Complete for **{sym_name}**")
    
    # Overall summary
    st.subheader("📊 Overall Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate average iota
    iotas = [ar_stats['iota'], sh_stats['iota'], cr_stats['iota'], so_stats['iota']]
    finite_iotas = [i for i in iotas if np.isfinite(i)]
    avg_iota = np.mean(finite_iotas) if finite_iotas else 0
    avg_rating = iota_to_persistence_rating(avg_iota)
    
    with col1:
        st.metric("Average Iota", f"{avg_iota:+.3f}")
    with col2:
        st.metric("Average Rating", f"{avg_rating}")
    with col3:
        st.metric("Reliability", reliability.replace("_", " "))
    with col4:
        sig_count = sum([ar_stats['significant'], sh_stats['significant'], 
                        cr_stats['significant'], so_stats['significant']])
        st.metric("Significant Metrics", f"{sig_count}/4")
    
    # Overall interpretation
    interpretation = interpret_iota_directly(avg_iota)
    if avg_iota >= 0.5:
        st.markdown(f'<div class="success-card"><strong>Overall Assessment:</strong> {interpretation}</div>', 
                   unsafe_allow_html=True)
    elif avg_iota >= -0.5:
        st.markdown(f'<div class="metric-card"><strong>Overall Assessment:</strong> {interpretation}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-card"><strong>Overall Assessment:</strong> {interpretation}</div>', 
                   unsafe_allow_html=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Metric Analysis")
    
    metrics_data = [
        ("Annualized Return", ar_stats, ar_oos, lambda x: f"{x*100:.2f}%"),
        ("Sharpe Ratio", sh_stats, sh_oos, lambda x: f"{x:.3f}"),
        ("Cumulative Return", cr_stats, cr_oos, lambda x: f"{x*100:.2f}%"),
        ("Sortino Ratio", so_stats, so_oos, format_sortino_output)
    ]
    
    for metric_name, stats_dict, oos_val, formatter in metrics_data:
        with st.expander(f"📊 {metric_name}", expanded=True):
            display_metric_detail(metric_name, stats_dict, oos_val, formatter)
    
    # Download section
    st.subheader("💾 Download Results")
    
    # Create summary for download
    summary_text = create_summary_text(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                                     ar_oos, sh_oos, cr_oos, so_oos, reliability, config)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"{sym_name}_iota_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Create CSV data
        csv_data = create_csv_data(ar_stats, sh_stats, cr_stats, so_stats, 
                                  ar_oos, sh_oos, cr_oos, so_oos)
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data,
            file_name=f"{sym_name}_iota_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_metric_detail(metric_name, stats_dict, oos_val, formatter):
    """Display detailed analysis for a single metric."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("IS Median", formatter(stats_dict['median_is']))
    with col2:
        st.metric("OOS Actual", formatter(oos_val))
    with col3:
        iota = stats_dict['iota']
        st.metric("Iota (ι)", f"{iota:+.3f}")
    with col4:
        st.metric("Persistence Rating", f"{stats_dict['persistence_rating']}")
    
    # Interpretation
    interpretation = interpret_iota_directly(stats_dict['iota'])
    if stats_dict['iota'] >= 0.5:
        st.success(f"**{interpretation}**")
    elif stats_dict['iota'] >= -0.5:
        st.info(f"**{interpretation}**")
    else:
        st.warning(f"**{interpretation}**")
    
    # Statistical details
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence interval
        ci_lower, ci_upper = stats_dict['confidence_interval']
        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
            st.write(f"**95% Confidence Interval:** [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # IQR
        q25, q75 = stats_dict['iqr_is']
        st.write(f"**IS Range (25th-75th):** {formatter(q25)} - {formatter(q75)}")
    
    with col2:
        # P-value and significance
        if np.isfinite(stats_dict['p_value']):
            sig_marker = " ***" if stats_dict['significant'] else ""
            st.write(f"**P-value:** {stats_dict['p_value']:.3f}{sig_marker}")
            if stats_dict['significant']:
                st.write("✅ **Statistically significant**")
            else:
                st.write("ℹ️ Not statistically significant")

def create_summary_text(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                       ar_oos, sh_oos, cr_oos, so_oos, reliability, config):
    """Create a summary report for download."""
    
    pc = lambda x: f"{x*100:.2f}%"
    
    summary = f"""IOTA CALCULATOR ANALYSIS SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

STRATEGY: {sym_name}
COMPOSER URL: {config['url']}

ANALYSIS CONFIGURATION
======================
Data Period: {config['early_date']} to {config['today_date']}
OOS Start Date: {config['oos_start']}
Number of IS Slices: {config['n_slices']}
Overlapping Slices: {'Yes' if config['overlap'] else 'No'}
Reliability Level: {reliability}

PERFORMANCE SUMMARY
===================
                    IS Median    OOS Actual    Iota (ι)    Rating    Significant
Annualized Return   {pc(ar_stats['median_is']):>10}   {pc(ar_oos):>10}   {ar_stats['iota']:>+7.3f}   {ar_stats['persistence_rating']:>6}   {'***' if ar_stats['significant'] else '   '}
Sharpe Ratio        {sh_stats['median_is']:>10.3f}   {sh_oos:>10.3f}   {sh_stats['iota']:>+7.3f}   {sh_stats['persistence_rating']:>6}   {'***' if sh_stats['significant'] else '   '}
Cumulative Return   {pc(cr_stats['median_is']):>10}   {pc(cr_oos):>10}   {cr_stats['iota']:>+7.3f}   {cr_stats['persistence_rating']:>6}   {'***' if cr_stats['significant'] else '   '}
Sortino Ratio       {format_sortino_output(so_stats['median_is']):>10}   {format_sortino_output(so_oos):>10}   {so_stats['iota']:>+7.3f}   {so_stats['persistence_rating']:>6}   {'***' if so_stats['significant'] else '   '}

DETAILED ANALYSIS
================="""
    
    metrics_data = [
        ("Annualized Return", ar_stats, ar_oos, pc),
        ("Sharpe Ratio", sh_stats, sh_oos, lambda x: f"{x:.3f}"),
        ("Cumulative Return", cr_stats, cr_oos, pc),
        ("Sortino Ratio", so_stats, so_oos, format_sortino_output)
    ]
    
    for metric_name, stats_dict, oos_val, formatter in metrics_data:
        summary += f"""

{metric_name.upper()}:
  IS Median: {formatter(stats_dict['median_is'])}
  OOS Actual: {formatter(oos_val)}
  Iota (ι): {stats_dict['iota']:+.3f}
  Persistence Rating: {stats_dict['persistence_rating']}
  Interpretation: {interpret_iota_directly(stats_dict['iota'])}
  
  95% Confidence Interval: [{stats_dict['confidence_interval'][0]:.3f}, {stats_dict['confidence_interval'][1]:.3f}]
  P-value: {stats_dict['p_value']:.3f}
  Statistically Significant: {'Yes' if stats_dict['significant'] else 'No'}
"""
    
    # Overall assessment
    iotas = [ar_stats['iota'], sh_stats['iota'], cr_stats['iota'], so_stats['iota']]
    finite_iotas = [i for i in iotas if np.isfinite(i)]
    avg_iota = np.mean(finite_iotas) if finite_iotas else 0
    avg_rating = iota_to_persistence_rating(avg_iota)
    
    summary += f"""

OVERALL ASSESSMENT
==================
Average Iota: {avg_iota:+.3f}
Average Persistence Rating: {avg_rating}
Overall Interpretation: {interpret_iota_directly(avg_iota)}

Significant Metrics: {sum([ar_stats['significant'], sh_stats['significant'], cr_stats['significant'], so_stats['significant']])}/4
Reliability: {reliability}

INTERPRETATION GUIDE
====================
• Iota (ι) measures standard deviations from historical median
• Positive ι = Better OOS performance than expected
• Negative ι = Worse OOS performance than expected
• |ι| ≥ 1.0 = Major difference (>1 standard deviation)
• |ι| < 0.1 = Minimal difference (within noise)

• Persistence Rating: 0-500 scale
• 100 = Neutral (matches expectations)
• >100 = Outperformance
• <100 = Underperformance

• *** = Statistically significant (p < 0.05)

Generated by Iota Calculator - Streamlit Version
"""
    
    return summary

def create_csv_data(ar_stats, sh_stats, cr_stats, so_stats, 
                   ar_oos, sh_oos, cr_oos, so_oos):
    """Create CSV data for download."""
    
    data = []
    
    metrics = [
        ("Annualized_Return", ar_stats, ar_oos),
        ("Sharpe_Ratio", sh_stats, sh_oos),
        ("Cumulative_Return", cr_stats, cr_oos),
        ("Sortino_Ratio", so_stats, so_oos)
    ]
    
    for metric_name, stats_dict, oos_val in metrics:
        data.append({
            'Metric': metric_name,
            'IS_Median': stats_dict['median_is'],
            'OOS_Value': oos_val,
            'Iota': stats_dict['iota'],
            'Persistence_Rating': stats_dict['persistence_rating'],
            'CI_Lower': stats_dict['confidence_interval'][0],
            'CI_Upper': stats_dict['confidence_interval'][1],
            'P_Value': stats_dict['p_value'],
            'Significant': stats_dict['significant'],
            'Interpretation': interpret_iota_directly(stats_dict['iota'])
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def show_help_content():
    """Show help and documentation."""
    
    st.header("📚 How to Use the Iota Calculator")
    
    st.markdown("""
    ## What is the Iota Calculator?
    
    The **Iota Calculator** helps you understand whether your trading strategy is performing as expected 
    based on historical patterns. It answers the key question: *"Is my strategy actually working, 
    or did I just get lucky in my backtest?"*
    
    ## Step-by-Step Guide
    
    ### 1. 🔗 Get Your Composer Symphony URL
    - Log into Composer
    - Open your symphony
    - Copy the full URL from your browser
    - Paste it into the "Composer Symphony URL" field
    
    ### 2. 📅 Set Your Out-of-Sample Date
    **This is the most important setting!**
    
    - Choose the date when you started "live trading" or when your backtest ended
    - Everything **before** this date = historical backtest data
    - Everything **after** this date = "real world" performance
    - Example: If you started trading the strategy on Jan 1, 2022, set OOS start to 2022-01-01
    
    ### 3. ⚙️ Configure Analysis Parameters
    - **Number of IS Slices**: How many historical periods to compare (100 is good default)
    - **Overlapping Slices**: Keep this True for better statistics
    - **Exclusion Windows**: Optional - exclude market crashes or unusual periods
    
    ### 4. 🚀 Run the Analysis
    - Click "Run Iota Analysis"
    - Wait for the analysis to complete (may take 1-2 minutes)
    - View results in the "Results" tab
    
    ## Understanding Your Results
    
    ### 🎯 Iota (ι) Score
    **The main number that tells you how your strategy is doing:**
    
    - **ι = +1.0**: You're doing 1 standard deviation BETTER than expected ✅
    - **ι = 0.0**: You're performing exactly as expected ➡️
    - **ι = -1.0**: You're doing 1 standard deviation WORSE than expected ⚠️
    
    ### 📊 Persistence Rating
    **Easy-to-understand 0-500 scale:**
    
    - **100**: Neutral performance (matches expectations)
    - **>100**: Outperforming expectations
    - **<100**: Underperforming expectations
    
    ### 📈 Statistical Significance
    **The *** markers mean:**
    - **P-value < 0.05**: The difference is statistically significant
    - **No asterisks**: Could be due to random chance
    
    ## Example Interpretation
    
    **Scenario**: Your strategy historically got 15% annual returns. In the last year, you got 25%.
    
    **What Iota Analysis Shows**:
    1. Looks at 100 historical 1-year periods
    2. Finds you typically got 5% to 25% returns
    3. Calculates that 25% is normal (Iota ≈ +0.3)
    4. **Conclusion**: "Your strategy is working fine, you just had a good year"
    
    **VS. if you got 50% returns**:
    1. Same historical analysis
    2. 50% is way higher than you've EVER done (Iota ≈ +3.0)
    3. **Conclusion**: "Either incredible luck, or market conditions changed dramatically"
    
    ## Common Issues & Solutions
    
    ### ❌ "Missing dependencies" error
    **Solution**: Make sure you have:
    - `sim.py` file in the same folder as this app
    - All required packages installed (scipy, pandas, numpy, etc.)
    
    ### ❌ "Insufficient data" error
    **Solution**: 
    - Use longer time periods (need at least 30 days in each period)
    - Move your OOS start date to have more historical data
    
    ### ❌ "Could not fetch data" error
    **Solution**:
    - Check that your Composer URL is correct and public
    - Make sure the symphony has sufficient historical data
    
    ## Tips for Best Results
    
    1. **Use at least 2 years of historical data** before your OOS start
    2. **Use at least 6 months of OOS data** for meaningful results
    3. **Be honest about your OOS start date** - don't cherry-pick
    4. **Consider excluding major market crashes** if they're not representative
    5. **Look at multiple metrics** - don't rely on just one
    
    ## What the Metrics Mean
    
    - **Annualized Return**: Yearly return percentage
    - **Sharpe Ratio**: Return per unit of risk (higher is better)
    - **Cumulative Return**: Total return over the period
    - **Sortino Ratio**: Return per unit of downside risk (like Sharpe, but only counts bad volatility)
    
    ## Need More Help?
    
    If you're still confused:
    1. Try the example configuration first
    2. Start with a simple strategy you understand
    3. Compare results with what you expected
    4. Remember: This tool shows you *how unusual* your recent performance is, not whether it's "good" or "bad"
    """)
    
    # Add troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Error Messages
        
        **"Cannot import helpers from sim.py"**
        - Make sure `sim.py` is in the same directory as this Streamlit app
        - Check that `sim.py` contains the functions `fetch_backtest` and `calculate_portfolio_returns`
        
        **"Missing dependencies: scipy"**
        - Run: `pip install scipy`
        - Make sure your requirements.txt includes all necessary packages
        
        **"Could not form any IS slice of required length"**
        - Your historical period is too short
        - Reduce the number of slices or extend your data start date
        
        **"Analysis failed" with network errors**
        - Check your internet connection
        - Verify the Composer URL is accessible
        - Try again in a few minutes (sometimes Composer is slow)
        
        **"Invalid date format" or date-related errors**
        - Make sure your OOS start date is between your data start and end dates
        - Use the date pickers rather than typing dates manually
        - Ensure you have sufficient data in both IS and OOS periods
        
        **Analysis runs but shows strange results**
        - Check that your OOS start date is correct (this is critical!)
        - Verify that your symphony URL points to the right strategy
        - Consider excluding major market disruptions if they skew results
        
        ### Performance Tips
        
        **Analysis is slow:**
        - Reduce number of IS slices (try 50 instead of 100)
        - Use shorter date ranges for testing
        - Overlapping slices take longer but give better statistics
        
        **Need more reliable results:**
        - Use longer time periods (more data = better statistics)
        - Increase number of IS slices for better confidence intervals
        - Make sure your OOS period is at least 6 months long
        """)
    
    # Add footer with version info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
        <p><strong>Iota Calculator - Streamlit Version</strong></p>
        <p>Compatible with your existing sim.py • Built for Composer Symphony analysis</p>
        <p>For questions about methodology, see the Help tab</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
