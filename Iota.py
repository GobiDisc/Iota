#!/usr/bin/env python3
"""
Iota Calculator with Core and Rolling Window Analysis
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
import urllib.parse

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Iota Calculator Enhanced",
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

# ===== CORE FUNCTIONS (Enhanced from Iota.py) =====

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
    elif iota_val >= 0.25:
        return "📈 SLIGHT_IMPROVEMENT: OOS mildly above IS median"
    elif iota_val >= -0.25:
        return "🎯 OOS closely matches backtest"
    elif iota_val >= -0.5:
        return "📉 OOS slightly below IS median"
    elif iota_val >= -1.0:
        return "🚨 WARNING: OOS >0.5σ below IS median"
    elif iota_val >= -2.0:
        return "🔴 ALERT: OOS >1σ below IS median"
    else:
        return "💀 CRITICAL: OOS >2σ below IS median"

def calculate_autocorrelation_adjustment(values: np.ndarray, overlap: bool) -> float:
    """Calculate autocorrelation adjustment factor for overlapping slice p-values."""
    if not overlap or len(values) < 3:
        return 1.0
    
    try:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        
        if not np.isfinite(autocorr):
            autocorr = 0.0
        autocorr = np.clip(autocorr, -0.99, 0.99)
        
        n = len(values)
        if autocorr > 0:
            effective_n = n * (1 - autocorr) / (1 + autocorr)
            adjustment_factor = np.sqrt(effective_n / n)
        else:
            adjustment_factor = 1.0
        
        adjustment_factor = np.clip(adjustment_factor, 0.1, 1.0)
        return adjustment_factor
        
    except Exception:
        return 0.7

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
                      lower_is_better: bool = False, overlap: bool = True) -> Tuple[float, bool]:
    """Wilcoxon test with autocorrelation adjustment."""
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
        _, p_value_raw = stats.wilcoxon(slice_iotas, alternative='two-sided')
        
        if overlap:
            autocorr_adjustment = calculate_autocorrelation_adjustment(np.array(slice_iotas), overlap)
            p_value_adjusted = min(1.0, p_value_raw / autocorr_adjustment)
            return p_value_adjusted, p_value_adjusted < 0.05
        else:
            return p_value_raw, p_value_raw < 0.05
            
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
                                            lower_is_better=lower_is_better, overlap=overlap)
    
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

# ===== ROLLING WINDOW ANALYSIS FUNCTIONS =====

def smooth_iotas(iotas, window=3):
    """Apply rolling average smoothing to iota series."""
    if len(iotas) < window:
        return iotas
    
    smoothed = []
    for i in range(len(iotas)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        smoothed.append(np.mean(iotas[start_idx:end_idx]))
    return smoothed

def rolling_oos_analysis(daily_ret: pd.Series, oos_start_dt: date, 
                        is_ret: pd.Series, n_slices: int = 100, overlap: bool = True,
                        window_size: int = None, step_size: int = None, 
                        min_windows: int = 6) -> Dict[str, Any]:
    """Simplified rolling analysis for overfitting detection."""
    oos_data = daily_ret[daily_ret.index >= oos_start_dt]
    total_oos_days = len(oos_data)
    
    if total_oos_days > 1500:
        oos_data = oos_data.iloc[-1000:]
        total_oos_days = len(oos_data)
    
    if total_oos_days < 90:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Adaptive window sizing
    if window_size is None:
        if total_oos_days >= 504:
            window_size = 126
        elif total_oos_days >= 252:
            window_size = 84
        elif total_oos_days >= 189:
            window_size = 63
        else:
            window_size = max(21, total_oos_days // 4)
    
    if step_size is None:
        step_size = max(5, window_size // 8)
    
    if total_oos_days < window_size + step_size:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    max_possible_windows = (total_oos_days - window_size) // step_size + 1
    max_windows = min(60, max_possible_windows)
    
    # Create IS slices
    is_slices = build_slices(is_ret, window_size, n_slices, overlap)
    if not is_slices:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_IS_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Pre-compute IS metrics
    is_metrics = {
        'sh': [sharpe_ratio(s) for s in is_slices], 
        'cr': [cumulative_return(s) for s in is_slices],
        'so': [sortino_ratio(s) for s in is_slices]
    }
    
    # Create rolling windows
    windows = []
    start_idx = 0
    window_count = 0
    
    while start_idx + window_size <= len(oos_data) and window_count < max_windows:
        window_data = oos_data.iloc[start_idx:start_idx + window_size]
        if len(window_data) == window_size:
            window_num = len(windows) + 1
            
            window_sh = sharpe_ratio(window_data)
            window_cr = cumulative_return(window_data)
            window_so = sortino_ratio(window_data)
            
            window_iotas = {}
            for metric in ['sh', 'cr', 'so']:
                is_values = np.array(is_metrics[metric])
                oos_value = {'sh': window_sh, 'cr': window_cr, 'so': window_so}[metric]
                lower_is_better = False
                
                if len(is_values) > 0 and np.isfinite(oos_value):
                    iota = compute_iota(np.median(is_values), oos_value, window_size, 
                                      lower_is_better=lower_is_better, is_values=is_values)
                    window_iotas[metric] = iota
                else:
                    window_iotas[metric] = np.nan
            
            windows.append({
                'start_date': window_data.index[0],
                'end_date': window_data.index[-1],
                'window_num': window_num,
                'returns': window_data,
                'metrics': {
                    'sh': window_sh, 
                    'cr': window_cr,
                    'so': window_so
                },
                'iotas': window_iotas
            })
            window_count += 1
        start_idx += step_size
    
    if len(windows) < min_windows:
        return {
            'sufficient_data': False,
            'n_windows': len(windows),
            'overfitting_risk': 'INSUFFICIENT_WINDOWS',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Extract iota series for analysis
    window_nums = np.array([w['window_num'] for w in windows])
    
    metric_iotas = {}
    for metric in ['sh', 'cr', 'so']:
        metric_iotas[metric] = np.array([w['iotas'][metric] for w in windows if np.isfinite(w['iotas'][metric])])
    
    # Calculate trend slopes
    metric_slopes = {}
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) >= 3:
            try:
                slope, _, _, _, _ = stats.linregress(window_nums[:len(metric_iotas[metric])], metric_iotas[metric])
                metric_slopes[f'{metric}_slope'] = slope
            except:
                metric_slopes[f'{metric}_slope'] = np.nan
        else:
            metric_slopes[f'{metric}_slope'] = np.nan

    valid_slopes = [slope for slope in metric_slopes.values() if np.isfinite(slope)]
    avg_trend_slope = np.mean(valid_slopes) if valid_slopes else np.nan
    
    # Calculate degradation score
    degradation_score = 0

    all_iotas = []
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 0:
            all_iotas.extend(metric_iotas[metric])

    if len(all_iotas) > 0:
        all_iotas = np.array(all_iotas)
        
        avg_iota = np.mean(all_iotas)
        if avg_iota < -1.5:
            degradation_score += 4
        elif avg_iota < -1.0:
            degradation_score += 3
        elif avg_iota < -0.5:
            degradation_score += 2
        elif avg_iota < -0.2:
            degradation_score += 1
        
        negative_proportion = np.mean(all_iotas < 0)
        if negative_proportion > 0.9:
            degradation_score += 3
        elif negative_proportion > 0.75:
            degradation_score += 2
        elif negative_proportion > 0.6:
            degradation_score += 1
        
        severely_negative = np.mean(all_iotas < -1.0)
        if severely_negative > 0.5:
            degradation_score += 3
        elif severely_negative > 0.3:
            degradation_score += 2
        elif severely_negative > 0.1:
            degradation_score += 1

    # Calculate proportion of time spent under -0.5 iota for each metric
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 0:
            under_threshold_proportion = np.mean(np.array(metric_iotas[metric]) < -0.5)
            if under_threshold_proportion > 0.8:
                degradation_score += 4
            elif under_threshold_proportion > 0.6:
                degradation_score += 3
            elif under_threshold_proportion > 0.4:
                degradation_score += 2
            elif under_threshold_proportion > 0.2:
                degradation_score += 1
    
    # Get sophisticated risk assessment
    sophisticated_analysis = analyze_sophisticated_decay_risk({
        'sufficient_data': True,
        'metric_iotas': metric_iotas
    })
    
    if 'error' not in sophisticated_analysis:
        risk_level = sophisticated_analysis['overall_risk']
        degradation_score = sophisticated_analysis['total_risk_score']
    else:
        # Fallback to original method
        if degradation_score >= 12:
            risk_level = "CRITICAL"
        elif degradation_score >= 8:
            risk_level = "HIGH"
        elif degradation_score >= 5:
            risk_level = "MODERATE"
        elif degradation_score >= 2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
    return {
        'sufficient_data': True,
        'n_windows': len(windows),
        'windows': windows,
        'iota_trend_slope': avg_trend_slope,
        'metric_slopes': metric_slopes,
        'degradation_score': degradation_score,
        'overfitting_risk': risk_level,
        'window_size_days': window_size,
        'step_size_days': step_size,
        'total_oos_days': total_oos_days,
        'is_slices_used': len(is_slices),
        'metric_iotas': metric_iotas
    }

def analyze_sophisticated_decay_risk(rolling_results: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced decay risk analysis with time spent above/below zero and area integration."""
    if not rolling_results.get('sufficient_data', False):
        return {"error": "Insufficient data for analysis"}
    
    metric_iotas = rolling_results.get('metric_iotas', {})
    analysis = {}
    
    for metric_name, metric_data in metric_iotas.items():
        if len(metric_data) == 0:
            continue
            
        iotas = np.array(metric_data)
        
        # Time-based analysis
        time_above_zero = np.mean(iotas > 0)
        time_below_zero = np.mean(iotas < 0)
        time_at_zero = np.mean(iotas == 0)
        
        # Magnitude analysis
        positive_iotas = iotas[iotas > 0]
        negative_iotas = iotas[iotas < 0]
        
        avg_positive_magnitude = np.mean(positive_iotas) if len(positive_iotas) > 0 else 0
        avg_negative_magnitude = np.mean(negative_iotas) if len(negative_iotas) > 0 else 0
        max_positive = np.max(positive_iotas) if len(positive_iotas) > 0 else 0
        max_negative = np.min(negative_iotas) if len(negative_iotas) > 0 else 0
        
        # Area integration (simplified as sum of values)
        positive_area = np.sum(positive_iotas) if len(positive_iotas) > 0 else 0
        negative_area = np.sum(negative_iotas) if len(negative_iotas) > 0 else 0
        net_area = positive_area + negative_area
        
        # Risk scoring
        risk_score = 0
        
        # Time below zero penalty
        if time_below_zero > 0.8:
            risk_score += 4
        elif time_below_zero > 0.6:
            risk_score += 3
        elif time_below_zero > 0.4:
            risk_score += 2
        elif time_below_zero > 0.2:
            risk_score += 1
        
        # Magnitude penalty for negative performance
        if avg_negative_magnitude < -1.5:
            risk_score += 4
        elif avg_negative_magnitude < -1.0:
            risk_score += 3
        elif avg_negative_magnitude < -0.5:
            risk_score += 2
        elif avg_negative_magnitude < -0.2:
            risk_score += 1
        
        # Area imbalance penalty
        area_ratio = abs(negative_area / positive_area) if positive_area != 0 else float('inf')
        if area_ratio > 3.0:
            risk_score += 3
        elif area_ratio > 2.0:
            risk_score += 2
        elif area_ratio > 1.5:
            risk_score += 1
        
        # Consistency penalty (high variance in negative performance)
        if len(negative_iotas) > 1:
            negative_std = np.std(negative_iotas)
            if negative_std > 1.0:
                risk_score += 2
            elif negative_std > 0.5:
                risk_score += 1
        
        analysis[metric_name] = {
            'time_above_zero': time_above_zero,
            'time_below_zero': time_below_zero,
            'time_at_zero': time_at_zero,
            'avg_positive_magnitude': avg_positive_magnitude,
            'avg_negative_magnitude': avg_negative_magnitude,
            'max_positive': max_positive,
            'max_negative': max_negative,
            'positive_area': positive_area,
            'negative_area': negative_area,
            'net_area': net_area,
            'area_ratio': area_ratio if positive_area != 0 else float('inf'),
            'risk_score': risk_score
        }
    
    # Overall risk assessment
    total_risk_score = sum(metric['risk_score'] for metric in analysis.values())
    
    if total_risk_score >= 15:
        overall_risk = "CRITICAL"
    elif total_risk_score >= 10:
        overall_risk = "HIGH"
    elif total_risk_score >= 6:
        overall_risk = "MODERATE"
    elif total_risk_score >= 3:
        overall_risk = "LOW"
    else:
        overall_risk = "MINIMAL"
    
    return {
        'metric_analysis': analysis,
        'total_risk_score': total_risk_score,
        'overall_risk': overall_risk
    }

def interpret_overfitting_risk(rolling_results: Dict[str, Any]) -> str:
    """Generate sophisticated interpretation of rolling analysis results."""
    if not rolling_results.get('sufficient_data', False):
        return "Insufficient data for rolling analysis (need longer OOS period)"
    
    # Get sophisticated analysis
    decay_analysis = analyze_sophisticated_decay_risk(rolling_results)
    
    if 'error' in decay_analysis:
        return decay_analysis['error']
    
    n_windows = rolling_results['n_windows']
    overall_risk = decay_analysis['overall_risk']
    total_risk_score = decay_analysis['total_risk_score']
    
    interpretation = f"**Advanced Decay Risk Analysis** ({n_windows} windows)\n\n"
    interpretation += f"**Overall Risk Level**: {overall_risk} (Score: {total_risk_score})\n\n"
    
    # Add metric-specific insights
    for metric_name, analysis in decay_analysis['metric_analysis'].items():
        metric_display = {
            'sh': 'Sharpe Ratio',
            'cr': 'Cumulative Return', 
            'so': 'Sortino Ratio'
        }.get(metric_name, metric_name)
        
        interpretation += f"**{metric_display}**:\n"
        interpretation += f"  • Time above zero: {analysis['time_above_zero']:.1%}\n"
        interpretation += f"  • Time below zero: {analysis['time_below_zero']:.1%}\n"
        interpretation += f"  • Avg positive magnitude: {analysis['avg_positive_magnitude']:+.2f}\n"
        interpretation += f"  • Avg negative magnitude: {analysis['avg_negative_magnitude']:+.2f}\n"
        interpretation += f"  • Area ratio (neg/pos): {analysis['area_ratio']:.2f}\n"
        interpretation += f"  • Risk score: {analysis['risk_score']}\n\n"
    
    return interpretation

def create_distribution_histograms(ar_is_values, sh_is_values, cr_is_values, so_is_values, ar_oos, sh_oos, cr_oos, so_oos, symphony_name: str, n_oos_days: int = None):
    """Create histogram plots showing in-sample distributions with OOS values marked."""
    
    # Import make_subplots
    from plotly.subplots import make_subplots
    
    # Define metrics and their data with proper scaling
    metrics_data = [
        ("Annualized Return", ar_is_values, ar_oos, lambda x: f"{x*100:+.2f}%", "Annualized Return (%)"),
        ("Sharpe Ratio", sh_is_values, sh_oos, lambda x: f"{x:+.3f}", "Sharpe Ratio"),
        ("Cumulative Return", cr_is_values, cr_oos, lambda x: f"{x*100:+.2f}%", "Cumulative Return (%)"),
        ("Sortino Ratio", so_is_values, so_oos, format_sortino_output, "Sortino Ratio")
    ]
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create subplots - 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric[0] for metric in metrics_data],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add histograms for each metric
    for i, (metric_name, is_values, oos_val, formatter, axis_title) in enumerate(metrics_data):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Scale values for percentage metrics (Annualized Return and Cumulative Return)
        if metric_name in ["Annualized Return", "Cumulative Return"]:
            scaled_is_values = is_values * 100
            scaled_oos_val = oos_val * 100
        else:
            scaled_is_values = is_values
            scaled_oos_val = oos_val
        
        # Create histogram for in-sample values
        fig.add_trace(
            go.Histogram(
                x=scaled_is_values,
                name=metric_name,
                nbinsx=40,
                opacity=0.7,
                marker_color=colors[i],
                showlegend=False,
                hovertemplate=f"<b>{metric_name}</b><br>" +
                             f"IS Value: %{{x}}{'%' if metric_name in ['Annualized Return', 'Cumulative Return'] else ''}<br>" +
                             "Count: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Calculate median for in-sample values
        median_val = np.median(is_values)
        if metric_name in ["Annualized Return", "Cumulative Return"]:
            scaled_median_val = median_val * 100
        else:
            scaled_median_val = median_val
        
        # Add vertical line for median value (blue dashed line)
        fig.add_vline(
            x=scaled_median_val,
            line_dash="dash",
            line_color="blue",
            line_width=2,
            annotation_text=f"Median: {formatter(median_val)}",
            annotation_position="top left",
            annotation=dict(
                font=dict(size=10, color="blue"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="blue",
                borderwidth=1
            ),
            row=row, col=col
        )
        
        # Add vertical line for OOS value (red dashed line)
        fig.add_vline(
            x=scaled_oos_val,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"OOS: {formatter(oos_val)}",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            ),
            row=row, col=col
        )
    
    # Create title with period information
    period_info = ""
    if n_oos_days:
        period_info = f" (OOS Period: {n_oos_days} days)"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symphony_name} - In-Sample Distributions with OOS Values{period_info}",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=900,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_rolling_analysis_plot(rolling_results: Dict[str, Any], symphony_name: str) -> go.Figure:
    """Create interactive Plotly plot for rolling analysis."""
    if not rolling_results.get('sufficient_data', False):
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for rolling analysis",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    windows = rolling_results.get('windows', [])
    if len(windows) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 windows for meaningful plot",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract data
    dates = [w['start_date'] for w in windows]
    
    # Create figure
    fig = go.Figure()
    
    # Add metric lines
    colors = {'sh': '#9467bd', 'cr': '#1f77b4', 'so': '#ff7f0e'}
    names = {'sh': 'Sharpe Ratio', 'cr': 'Cumulative Return', 'so': 'Sortino Ratio'}
    
    for metric in ['sh', 'cr', 'so']:
        metric_data = []
        metric_dates = []
        for i, (date, window) in enumerate(zip(dates, windows)):
            iota_val = window['iotas'][metric]
            if np.isfinite(iota_val):
                metric_data.append(iota_val)
                metric_dates.append(date)
        
        if len(metric_data) >= 3:
            metric_data_smooth = smooth_iotas(metric_data, window=3)
            fig.add_trace(go.Scatter(
                x=metric_dates,
                y=metric_data_smooth,
                mode='lines+markers',
                name=f'{names[metric]} Iota (smoothed)',
                line=dict(color=colors[metric], width=2),
                marker=dict(size=4)
            ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                  annotation_text="Neutral Performance", annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgreen", 
                  annotation_text="Overperformance (+0.5σ)", annotation_position="top right")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="lightcoral", 
                  annotation_text="Underperformance (-0.5σ)", annotation_position="bottom right")
    
    # Update layout
    n_windows = rolling_results.get('n_windows', 0)
    window_size = rolling_results.get('window_size_days', 0)
    
    title_text = f'{symphony_name} - Rolling Iota Analysis'
    subtitle_text = f'{n_windows} windows ({window_size}d each) | Smoothed trends'
    
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{subtitle_text}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Time Period (OOS)",
        yaxis_title="Iota (ι)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
    
    return fig

# ===== STREAMLIT APP =====

def main():
    
    # Initialize query parameters for sharing
    query_params = st.query_params
    
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
            color: #333;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .success-card {
            background-color: #d4edda;
            color: #155724;
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
        .critical-card {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📊 Iota Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; font-size: 1.5rem; color: #666; margin-bottom: 2rem;">Is your strategy\'s performance matching the backtest?</h2>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 Configuration", "🔢 Results", "📊 Distributions", "📈 Rolling Analysis", "📚 Help"])
    
    # Configuration Tab
    with tab1:
        st.header("Analysis Configuration")
        
        # Main configuration form
        with st.form("analysis_form"):
            st.subheader("📝 Required Information")
            
            # Symphony URL
            default_url = query_params.get("url", "")
            url = st.text_input(
                "Composer Symphony URL *",
                value=default_url,
                help="Enter the full URL of your Composer symphony",
                placeholder="https://app.composer.trade/symphony/..."
            )
            
            # Date configuration
            col1, col2 = st.columns(2)
            with col1:
                default_early_date = query_params.get("early_date", "2000-01-01")
                early_date = st.date_input(
                    "Data Start Date:",
                    value=date.fromisoformat(default_early_date),
                    help="How far back to fetch data. Will default to oldest possible date if left untouched."
                )
            
            with col2:
                default_today_date = query_params.get("today_date", date.today().isoformat())
                today_date = st.date_input(
                    "Data End Date:",
                    value=date.fromisoformat(default_today_date),
                    help="End date for data fetching. Will default to most recent possible date if left untouched."
                )
            
            # OOS start date - this is crucial
            default_oos_start = query_params.get("oos_start", (date.today() - timedelta(days=730)).isoformat())
            oos_start = st.date_input(
                "Out-of-Sample Start Date *",
                value=date.fromisoformat(default_oos_start),
                help="⚠️ CRITICAL: Date when your 'live trading' or out-of-sample period begins. Everything before this is historical backtest data, everything after is 'real world' performance. This will NOT default to the OOS date indicated by Composer."
            )
            
            st.markdown("---")
            st.subheader("⚙️ Analysis Parameters")
            
            # Analysis parameters in columns
            col1, col2 = st.columns(2)
            with col1:
                default_n_slices = int(query_params.get("n_slices", "100"))
                n_slices = st.number_input(
                    "Number of IS Slices:",
                    min_value=10,
                    max_value=500,
                    value=default_n_slices,
                    help="How many historical periods to compare against (more = better statistics, slower analysis)"
                )
            
            with col2:
                default_overlap = query_params.get("overlap", "true").lower() == "true"
                overlap = st.checkbox(
                    "Allow Overlapping Slices",
                    value=default_overlap,
                    help="Whether historical comparison periods can overlap (recommended: True for more data)"
                )
            
            # Rolling analysis parameters
            st.subheader("🔄 Rolling Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                default_enable_rolling = query_params.get("enable_rolling", "true").lower() == "true"
                enable_rolling = st.checkbox(
                    "Enable Rolling Window Analysis",
                    value=default_enable_rolling,
                    help="Perform overfitting detection using rolling windows"
                )
            
            with col2:
                if enable_rolling:
                    default_auto_window = query_params.get("auto_window", "true").lower() == "true"
                    auto_window = st.checkbox(
                        "Auto Window Size",
                        value=default_auto_window,
                        help="Automatically determine optimal window size based on OOS period length. If unchecked, you'll need to hit 'Run Iota Analysis' and then fill out your preferred window parameters"
                    )
                else:
                    auto_window = True
            
            # Show manual window settings when auto is disabled
            if enable_rolling and not auto_window:
                col1, col2 = st.columns(2)
                with col1:
                    default_window_size = int(query_params.get("window_size", "126"))
                    window_size = st.number_input(
                        "Window Size (days):",
                        min_value=21,
                        max_value=252,
                        value=default_window_size,
                        help="Size of each rolling window in days"
                    )
                with col2:
                    default_step_size = int(query_params.get("step_size", "21"))
                    step_size = st.number_input(
                        "Step Size (days):",
                        min_value=1,
                        max_value=63,
                        value=default_step_size,
                        help="Days between window starts"
                    )
            else:
                window_size = None
                step_size = None
            
            # Note about form behavior
            if enable_rolling and not auto_window:
                st.info("💡 **Note**: Manual window settings will be applied when you submit the form.")
            
            # Optional exclusion windows
            st.subheader("🚫 Exclusion Windows (Optional)")
            default_exclusions_str = query_params.get("exclusions_str", "")
            exclusions_str = st.text_area(
                "Exclude specific date ranges:",
                value=default_exclusions_str,
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
                    # Store in session state
                    st.session_state.analysis_config = {
                        'url': url,
                        'early_date': early_date,
                        'today_date': today_date,
                        'oos_start': oos_start,
                        'n_slices': n_slices,
                        'overlap': overlap,
                        'exclusions_str': exclusions_str,
                        'enable_rolling': enable_rolling,
                        'auto_window': auto_window,
                        'window_size': window_size,
                        'step_size': step_size
                    }
                    st.session_state.run_analysis = True
                    st.session_state.auto_switch_to_results = True
                    
                    # Update URL with current parameters for sharing
                    
                    params = {
                        "url": url,
                        "early_date": early_date.isoformat(),
                        "today_date": today_date.isoformat(),
                        "oos_start": oos_start.isoformat(),
                        "n_slices": str(n_slices),
                        "overlap": str(overlap).lower(),
                        "exclusions_str": exclusions_str,
                        "enable_rolling": str(enable_rolling).lower(),
                        "auto_window": str(auto_window).lower(),
                    }
                    if window_size is not None:
                        params["window_size"] = str(window_size)
                    if step_size is not None:
                        params["step_size"] = str(step_size)
                    
                    # Set query parameters in the URL
                    for k, v in params.items():
                        if v:  # Only include non-empty values
                            st.query_params[k] = str(v)
                    
                    # Create shareable URL for display
                    encoded_params = []
                    for k, v in params.items():
                        if v:  # Only include non-empty values
                            encoded_key = urllib.parse.quote(k)
                            encoded_value = urllib.parse.quote(str(v))
                            encoded_params.append(f"{encoded_key}={encoded_value}")
                    
                    query_string = "&".join(encoded_params)
                    base_url = "https://iotametrics.streamlit.app/"
                    shareable_url = f"{base_url}?{query_string}"
                    
                    # Store the shareable URL in session state for display outside the form
                    st.session_state.shareable_url = shareable_url
                    
                    # Success message with clear navigation instructions
                    st.success("✅ Configuration saved! Click the '📊 Results' tab above to view your analysis.")

        # Display shareable URL outside the form (if available)
        if hasattr(st.session_state, 'shareable_url') and st.session_state.shareable_url:
            st.markdown("---")
            st.markdown("### 🔗 Share Your Analysis")
            st.info(f"**Shareable URL**: Copy this link to share your analysis settings with others:")
            
            # Create a text area for easy copying
            st.text_area(
                "Shareable URL (select all and copy):",
                value=st.session_state.shareable_url,
                height=100,
                help="Select all text (Ctrl+A) then copy (Ctrl+C)",
                key="shareable_url_textarea"
            )
            






    # Results Tab
    with tab2:
                                 st.header("🔢 Core Iota Analysis Results")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
            config = st.session_state.analysis_config
            
            try:
                # Data preparation section
                st.markdown("### 🔄 Data Preparation")
                
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
                st.markdown("")  # Add spacing
                
                with st.spinner("🧮 Calculating portfolio returns..."):
                    # Capture stdout during portfolio calculation
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
                
                # Convert index to date
                daily_ret.index = pd.to_datetime(daily_ret.index).date
                
                st.info(f"📈 Loaded {len(daily_ret)} days of return data")
                st.markdown("")  # Add spacing
                
                # Apply exclusions
                if exclusions:
                    mask = pd.Series(True, index=daily_ret.index)
                    for s, e in exclusions:
                        mask &= ~((daily_ret.index >= s) & (daily_ret.index <= e))
                    removed = (~mask).sum()
                    daily_ret = daily_ret[mask]
                    st.warning(f"🚫 Excluded {removed} days across {len(exclusions)} window(s)")
                    st.markdown("")  # Add spacing
                
                # Data split section
                st.markdown("### 📊 Data Split Summary")
                
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
                
                st.markdown("---")  # Add divider before analysis
                st.markdown("### 🧮 Core Analysis")
                
                with st.spinner("📊 Running core Iota analysis..."):
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
                
                # Store results in session state for rolling analysis
                st.session_state.core_results = {
                    'sym_name': sym_name,
                    'daily_ret': daily_ret,
                    'is_ret': is_ret,
                    'oos_ret': oos_ret,
                    'ar_stats': ar_stats,
                    'sh_stats': sh_stats,
                    'cr_stats': cr_stats,
                    'so_stats': so_stats,
                    'ar_oos': ar_oos,
                    'sh_oos': sh_oos,
                    'cr_oos': cr_oos,
                    'so_oos': so_oos,
                    'reliability': reliability,
                    'n_slices': len(slices)
                }
                
                # Store core results in session state for other tabs
                st.session_state.core_results = {
                    'sym_name': sym_name,
                    'ar_stats': ar_stats,
                    'sh_stats': sh_stats,
                    'cr_stats': cr_stats,
                    'so_stats': so_stats,
                    'ar_oos': ar_oos,
                    'sh_oos': sh_oos,
                    'cr_oos': cr_oos,
                    'so_oos': so_oos,
                    'reliability': reliability,
                    'config': config,
                    'ar_is_values': df["ar_is"].values,
                    'sh_is_values': df["sh_is"].values,
                    'cr_is_values': df["cr_is"].values,
                    'so_is_values': df["so_is"].values
                }
                
                # Display core results
                display_core_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                                   ar_oos, sh_oos, cr_oos, so_oos, reliability, config)
                
                st.markdown("---")  # Add divider before rolling analysis
                
                # Run rolling analysis if enabled
                if config['enable_rolling']:
                    st.markdown("### 🔄 Rolling Window Analysis")
                    with st.spinner("🔄 Running rolling window analysis..."):
                        rolling_results = rolling_oos_analysis(
                            daily_ret, oos_start_dt, is_ret, 
                            config['n_slices'], config['overlap'],
                            config['window_size'], config['step_size']
                        )
                        st.session_state.rolling_results = rolling_results
                    
                    st.success("✅ Rolling analysis complete! Check the 'Rolling Analysis' tab for overfitting insights.")
                else:
                    st.info("ℹ️ Rolling analysis disabled. Enable it in the configuration to detect overfitting patterns.")
                
                # Reset the flag
                st.session_state.run_analysis = False
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Please configure and run your analysis in the 'Configuration' tab first.")

    # Distributions Tab
    with tab3:
        st.header("📊 In-Sample Distributions")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'core_results') and st.session_state.core_results:
            core_results = st.session_state.core_results
            
            st.success("✅ Distribution analysis ready!")
            st.markdown("")  # Add spacing
            
            # Display period information
            st.markdown("### 📅 Comparison Periods")
            if hasattr(core_results, 'config') and core_results['config']:
                config = core_results['config']
                oos_start = config.get('oos_start')
                today_date = config.get('today_date')
                if oos_start and today_date:
                    oos_days = (today_date - oos_start).days
                    st.info(f"**In-Sample Period**: Historical backtest data (multiple overlapping periods)  \n**Out-of-Sample Period**: {oos_start.strftime('%Y-%m-%d')} to {today_date.strftime('%Y-%m-%d')} ({oos_days} days)")
            
            # Display distribution histograms
            st.markdown("### 📈 Metric Distributions")
            st.markdown("Histograms show the distribution of in-sample values for each metric, with red dashed lines indicating where your out-of-sample values fall.")
            
            # Create and display histogram
            oos_days = None
            if hasattr(core_results, 'config') and core_results['config']:
                config = core_results['config']
                oos_start = config.get('oos_start')
                today_date = config.get('today_date')
                if oos_start and today_date:
                    oos_days = (today_date - oos_start).days
            
            fig = create_distribution_histograms(
                core_results['ar_is_values'], 
                core_results['sh_is_values'], 
                core_results['cr_is_values'], 
                core_results['so_is_values'],
                core_results['ar_oos'], 
                core_results['sh_oos'], 
                core_results['cr_oos'], 
                core_results['so_oos'],
                core_results['sym_name'],
                oos_days
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            st.markdown("### 🎯 Interpretation")
            st.markdown("""
            - **Red dashed lines**: Show where your OOS performance falls relative to the IS distribution
            - **Histogram bars**: Show the frequency of different performance levels during the backtest period
            - **Left of distribution**: OOS underperforming relative to backtest expectations
            - **Right of distribution**: OOS outperforming relative to backtest expectations
            - **Center of distribution**: OOS performance matches backtest expectations
            """)
        else:
            st.info("📊 No analysis data available. Please run the analysis first in the 'Results' tab.")

    # Rolling Analysis Tab
    with tab4:
        st.header("📈 Rolling Window Analysis")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'rolling_results') and st.session_state.rolling_results:
            rolling_results = st.session_state.rolling_results
            
            if rolling_results.get('sufficient_data', False):
                st.success("✅ Rolling analysis completed successfully!")
                st.markdown("")  # Add spacing
                
                # Display results
                st.markdown("### 📊 Analysis Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Windows", rolling_results['n_windows'])
                with col2:
                    st.metric("Window Size", f"{rolling_results['window_size_days']}d")
                with col3:
                    st.metric("Decay Risk", rolling_results['overfitting_risk'], 
                             help="Advanced risk assessment considering time spent above/below zero, magnitude of performance, area integration, and consistency. MINIMAL/LOW = good, MODERATE = concerning, HIGH/CRITICAL = likely overfit. [Updated]")
                
                st.markdown("")  # Add spacing
                
                # Display sophisticated decay risk analysis
                st.markdown("### 🧠 Advanced Decay Risk Analysis")
                interpretation = interpret_overfitting_risk(rolling_results)
                st.markdown(interpretation)
                
                st.markdown("")  # Add spacing
                
                # Create and display plot
                if hasattr(st.session_state, 'core_results'):
                    sym_name = st.session_state.core_results['sym_name']
                    
                    # Center the chart section header
                    st.markdown('<h3 style="text-align: center;">📈 Rolling Performance Chart</h3>', unsafe_allow_html=True)
                    fig = create_rolling_analysis_plot(rolling_results, sym_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show last window info (centered)
                    if rolling_results.get('windows'):
                        last_window = rolling_results['windows'][-1]
                        st.markdown("")  # Add spacing
                        st.markdown(f'<p style="text-align: center;"><strong>📅 Last Window</strong>: {last_window["start_date"]} to {last_window["end_date"]}</p>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Insufficient data for rolling analysis")
                st.write("**Recommendation**: Extend OOS period to at least 6 months for meaningful rolling analysis")
        else:
            st.info("📊 No rolling analysis data available. Please run the analysis first in the 'Results' tab with rolling analysis enabled.")

    # Help Tab
    with tab5:
        show_comprehensive_help()

def display_core_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                        ar_oos, sh_oos, cr_oos, so_oos, reliability, config):
    """Display the core analysis results."""
    
    # Header with success message
    st.success(f"🎉 Core Analysis Complete for **{sym_name}**")
    st.markdown("---")  # Add divider
    
    # Overall summary section
    st.subheader("📊 Overall Summary")
    st.markdown("")  # Add spacing
    
    # Metrics in columns
    col1, col2 = st.columns(2)
    
    # Calculate average iota
    iotas = [ar_stats['iota'], sh_stats['iota'], cr_stats['iota'], so_stats['iota']]
    finite_iotas = [i for i in iotas if np.isfinite(i)]
    avg_iota = np.mean(finite_iotas) if finite_iotas else 0
    avg_rating = iota_to_persistence_rating(avg_iota)
    
    with col1:
        st.metric("Composite Iota", f"{avg_iota:+.3f}", 
                 help="Average of all metric iota values. Measures how many standard deviations your overall OOS performance differs from backtest expectations. Positive = outperforming, negative = underperforming.")
    with col2:
        st.metric("Composite Persistence Rating", f"{avg_rating}", 
                 help="0-100+ scale rating of strategy persistence. 100 = strategy exactly matches backtest, <100 = strategy underperforms relative to backtest, >100 = strategy outperforms backtest.")
    
    st.markdown("")  # Add spacing after metrics
    
    # Overall interpretation with better spacing
    interpretation = interpret_iota_directly(avg_iota)
    st.markdown("### Overall Assessment")
    if avg_iota >= 0.5:
        st.markdown(f'<div class="success-card" style="font-size: 1.2rem;"><strong>{interpretation}</strong></div>', 
                   unsafe_allow_html=True)
    elif avg_iota >= 0.1:
        st.markdown(f'<div class="metric-card" style="font-size: 1.2rem;"><strong>{interpretation}</strong></div>', 
                   unsafe_allow_html=True)
    elif avg_iota >= -0.1:
        st.markdown(f'<div class="info-card" style="font-size: 1.2rem;"><strong>⚠️ {interpretation}</strong></div>', unsafe_allow_html=True)
    elif avg_iota >= -0.5:
        st.markdown(f'<div class="info-card" style="font-size: 1.2rem;"><strong>📉 {interpretation}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="critical-card" style="font-size: 1.2rem;"><strong>⚠️ {interpretation}</strong></div>', unsafe_allow_html=True)
    
    st.markdown("---")  # Add divider before detailed metrics
    
    # Detailed metrics section
    st.subheader("📈 Detailed Metric Analysis")
    st.markdown("")  # Add spacing
    
    metrics_data = [
        ("Annualized Return", ar_stats, ar_oos, lambda x: f"{x*100:.2f}%"),
        ("Sharpe Ratio", sh_stats, sh_oos, lambda x: f"{x:.3f}"),
        ("Cumulative Return", cr_stats, cr_oos, lambda x: f"{x*100:.2f}%"),
        ("Sortino Ratio", so_stats, so_oos, format_sortino_output)
    ]
    
    for i, (metric_name, stats_dict, oos_val, formatter) in enumerate(metrics_data):
        with st.expander(f"📊 {metric_name}", expanded=True):
            display_metric_detail(metric_name, stats_dict, oos_val, formatter)
        
        # Add spacing between expanders (except for the last one)
        if i < len(metrics_data) - 1:
            st.markdown("")  # Add spacing between metric sections

def display_metric_detail(metric_name, stats_dict, oos_val, formatter):
    """Display detailed analysis for a single metric."""
    
    # Key metrics section
    st.markdown("#### 📊 Key Metrics")
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
    
    st.markdown("")  # Add spacing
    
    # Interpretation section
    st.markdown("#### Interpretation")
    interpretation = interpret_iota_directly(stats_dict['iota'])
    if stats_dict['iota'] >= 0.5:
        st.success(f"**{interpretation}**")
    elif stats_dict['iota'] >= -0.5:
        st.info(f"**{interpretation}**")
    else:
        st.warning(f"**{interpretation}**")
    
    st.markdown("")  # Add spacing
    
    # Statistical details section
    st.markdown("#### 📈 Statistical Details")
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
        # Empty column for layout balance
        st.write("")

def show_comprehensive_help():
    """Show comprehensive help and documentation from the original Iota.py."""
    
    st.header("📚 Comprehensive Iota Calculator Guide")
    
    # Create help sub-tabs
    help_tab1, help_tab2, help_tab3, help_tab4, help_tab5, help_tab6 = st.tabs([
        "🎯 Quick Start", "🧮 Methodology", "📊 Interpretation", "📈 Distributions", "🔄 Rolling Analysis", "❓ FAQ"
    ])
    
    with help_tab1:
        st.subheader("🎯 Quick Start Guide")
        
        st.markdown("""
        ## What is the Iota Calculator?
        
        The **Iota Calculator** helps you understand whether your trading strategy is performing as expected 
        based on historical patterns. It answers the key question: *"Is my strategy's performance consistent with its backtest?"*
        
        ### Key Features:
        - 📊 **Core Iota Analysis**: Compare OOS performance to historical expectations
        - 📊 **Distribution Analysis**: Visualize in-sample distributions with OOS values
        - 🔄 **Rolling Window Analysis**: Advanced decay risk assessment with sophisticated metrics
        - 📈 **Interactive Visualizations**: Track performance trends with Plotly charts
        - 🎯 **Statistical Rigor**: Autocorrelation-adjusted analysis and confidence intervals
        
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
        - **Rolling Analysis**: Enable for advanced decay risk assessment
        - **Exclusion Windows**: Optional - exclude market crashes or unusual periods
        
        ### 4. 🚀 Run the Analysis
        - Click "Run Iota Analysis"
        - Wait for the analysis to complete (may take 2-3 minutes)
        - View core results in the "Results" tab
        - Explore distributions in the "Distributions" tab
        - Check rolling analysis in the "Rolling Analysis" tab
        
        ### 5. 🔗 Share Your Analysis
        - After running the analysis, a shareable URL will be generated
        - Copy the URL to share your exact configuration and results with others
        - Anyone with the link can view the same analysis settings and results
        - Perfect for team collaboration, peer review, or documentation
        
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

        ### 🔄 Decay Risk
        **Advanced rolling analysis shows if your strategy has degraded out of sample:**
        
        - **MINIMAL/LOW**: Strategy working as expected ✅
        - **MODERATE**: Some concerns, monitor closely ⚠️
        - **HIGH/CRITICAL**: Likely overfit and/or market conditions have substantially changed, running something else 🚨
        """)
    
    with help_tab2:
        st.subheader("🧮 Detailed Methodology")
        
        st.markdown("""
        ## WHAT IS IOTA (ι)?
        
        Iota is a standardized metric that measures how many standard deviations your out-of-sample performance differs from the in-sample median, adjusted for sample size.
        
        **Formula:** `ι = weight × (OOS_metric - IS_median) / IS_std_dev`
        
        Where:
        - `weight = min(1.0, √(OOS_days / 252))` accounts for sample size reliability
        - `OOS_metric` = your strategy's out-of-sample performance value
        - `IS_median` = median of all in-sample slice performances  
        - `IS_std_dev` = standard deviation of in-sample slice performances
        
        **INTERPRETATION:**
        - `ι = +1.0`: OOS performed 1 standard deviation BETTER than historical median
        - `ι = -1.0`: OOS performed 1 standard deviation WORSE than historical median
        - `ι = 0`: OOS performance matches historical expectations exactly
        - `|ι| ≥ 1.0`: Major difference (statistically significant)
        - `|ι| < 0.1`: Minimal difference (within noise)
        
        ## Step-by-Step Analysis Process
        
        ### Step 1: Data Preparation and Slice Construction
        **What happens:**
        1. Historical data is split into In-Sample (IS) and Out-of-Sample (OOS) periods at your specified date
        2. The IS period is divided into multiple overlapping or non-overlapping "slices"
        3. Each slice has the same length as your OOS period for fair comparison
        
        **Rationale:**
        - **Temporal consistency**: Each IS slice represents what your strategy would have done during a period of identical length to your actual OOS period
        - **Distribution building**: Multiple slices create a distribution of historical performance under similar conditions
        - **Avoiding look-ahead bias**: Only data prior to OOS start date is used for IS analysis
        
        ### Step 2: Metric Calculation
        **What happens:**
        1. Four core metrics calculated for OOS period: Annualized Return, Sharpe Ratio, Cumulative Return, Sortino Ratio
        2. Same metrics calculated for each IS slice
        3. Statistical distribution properties computed for IS metrics (median, standard deviation, quartiles)
        
        **Rationale:**
        - **Comprehensive coverage**: These metrics capture different aspects of strategy performance
        - **Risk-adjustment**: Sharpe and Sortino ratios account for volatility and downside risk
        - **Comparability**: Same metrics across all time periods enable direct comparison
        
        ### Step 3: Enhanced Iota Calculation
        **What happens:**
        ```
        ι = weight × (OOS_metric - IS_median) / IS_std_dev
        ```
        
        **Rationale:**
        - **Standardization**: Converts absolute differences to standard deviation units for universal interpretation
        - **Sample size weighting**: Longer OOS periods get more weight (up to 1 year = full weight)
        - **Robust statistics**: Median and standard deviation are less sensitive to outliers than mean
        - **Intuitive scale**: ι = +1.0 means OOS performed 1 standard deviation better than typical
        
        ### Step 4: Statistical Testing with Autocorrelation Adjustment
        **What happens:**
        1. **For overlapping slices**: Calculate first-order autocorrelation of IS values
        2. **Effective sample size**: Adjust for temporal correlation using Newey-West correction
        3. **P-value adjustment**: Scale p-values upward to account for reduced independence
        4. **Bootstrap confidence intervals**: Use block bootstrap for overlapping data, standard bootstrap for non-overlapping
        
        **Mathematical detail:**
        ```
        effective_n = n × (1 - autocorr) / (1 + autocorr)
        adjustment_factor = √(effective_n / n)
        p_value_adjusted = min(1.0, p_value_raw / adjustment_factor)
        ```
        
        ### Step 5: Distribution Analysis
        **What happens:**
        1. **Histogram creation**: In-sample distributions plotted for each metric
        2. **OOS value marking**: Red dashed line shows where OOS performance falls
        3. **Median reference**: Blue dashed line shows in-sample median
        4. **Visual comparison**: Easy to see OOS performance relative to historical distribution
        
        **Rationale:**
        - **Visual clarity**: Histograms make distribution shape and OOS position obvious
        - **Intuitive interpretation**: Left of distribution = underperforming, right = outperforming
        - **Median reference**: Shows expected performance level
        - **Multi-metric view**: All four metrics displayed simultaneously
        
        ### Step 6: Rolling Window Analysis (Advanced Decay Risk Assessment)
        **What happens:**
        1. **Window creation**: OOS period divided into overlapping windows (e.g., 6-month windows with 1-month steps)
        2. **Historical comparison**: Each window compared against IS slice distribution
        3. **Sophisticated analysis**: Time spent above/below zero, magnitude analysis, area integration
        4. **Advanced scoring**: Multiple factors including consistency and area imbalance
        
        **Rationale:**
        - **Overfitting detection**: Strategies that are overfit show declining performance over time
        - **Temporal granularity**: Rolling windows reveal when and how performance changes
        - **Early warning**: Identifies degradation before it becomes severe
        - **Comprehensive assessment**: Considers both frequency and severity of underperformance
        
        ## CORE METRICS ANALYZED
        
        1. **ANNUALIZED RETURN**: Yearly return percentage (CAGR)
        2. **SHARPE RATIO**: Risk-adjusted return measure (return per unit of total volatility)
        3. **CUMULATIVE RETURN**: Total return over the entire period
        4. **SORTINO RATIO**: Downside risk-adjusted return (return per unit of downside volatility)
        
        ## AUTOCORRELATION ADJUSTMENT
        
        When using overlapping slices, the temporal correlation between adjacent slices reduces the effective sample size and can lead to overly optimistic p-values.
        
        **This calculator automatically:**
        1. Detects overlapping slice configurations
        2. Calculates the first-order autocorrelation of IS slice metrics
        3. Adjusts the effective sample size using Newey-West type correction
        4. Provides more conservative, statistically valid p-values
        
        **The adjustment factor** is reported for transparency, typically ranging from 0.3-1.0:
        - `1.0` = No adjustment (non-overlapping slices)
        - `0.7` = Moderate positive autocorrelation typical of overlapping financial data
        - `0.3` = High positive autocorrelation requiring significant adjustment
        """)
    
    with help_tab3:
        st.subheader("📊 Interpretation Guide")
        
        st.markdown("""
        ## Understanding Iota Values
        
        | Iota Range | Rating Range | Interpretation | Action |
        |------------|--------------|----------------|---------|
        | **ι ≥ +2.0** | ~270+ | 🔥 **EXCEPTIONAL**: >2σ above median | Continue strategy, consider scaling |
        | **ι ≥ +1.0** | ~165+ | ✅ **EXCELLENT**: >1σ above median | Strong performance, monitor |
        | **ι ≥ +0.5** | ~128+ | 👍 **GOOD**: >0.5σ above median | Solid outperformance |
        | **ι ≥ +0.25** | ~113+ | 📈 **SLIGHT_IMPROVEMENT** | Mild improvement |
        | **-0.25 ≤ ι ≤ +0.25** | 88-113 | ➡️ **OOS closely matches backtest** | Performing as expected |
        | **ι ≤ -0.25** | ~88- | ⚠️ **CAUTION**: Below median | Monitor closely |
        | **ι ≤ -0.5** | ~78- | 🚨 **WARNING**: >0.5σ below | Consider adjustments |
        | **ι ≤ -1.0** | ~60- | 🔴 **ALERT**: >1σ below | Significant concern |
        | **ι ≤ -2.0** | ~36- | 💀 **CRITICAL**: >2σ below | Strategy likely failing |
        
        ## Persistence Ratings Explained
        
        The **Persistence Rating** converts iota (ι) into a more intuitive 0–500 point scale using:
        
        **Formula:** `Rating = 100 × exp(0.5 × ι)`
        
        ### 🧠 Key Insights:
        - **100** = Neutral baseline (matches historical expectations)
        - **>130** = Meaningful outperformance
        - **<80** = Concerning underperformance
        - **>200** = Exceptional performance (rare)
        - **<50** = Critical underperformance
        
        ### 🎯 Why Use Ratings?
        - **Non-technical summary** of complex statistical analysis
        - **Cross-strategy comparisons** easier (Rating 170 vs. Rating 90)
        - **Intuitive interpretation** without understanding standard deviations
        
        ## Statistical Significance and P-Values
        
        ### **What the P-Value Means**
        The p-value answers: *"If my strategy actually performed no differently than random historical periods, what's the probability I would see a difference this large or larger by pure chance?"*
        
        **Example interpretations:**
        - **p = 0.001**: Only 0.1% chance this difference is due to random luck
        - **p = 0.050**: 5% chance this difference is due to random luck  
        - **p = 0.200**: 20% chance this difference is due to random luck
        
        ### **Significance Markers**
        - ***** (3 asterisks) = p < 0.05 after autocorrelation adjustment = "statistically significant"
        - **No asterisks**: p ≥ 0.05 = difference could plausibly be due to random variation
        
        ### **Autocorrelation Adjustment Impact**
        When you see autocorrelation adjustment factors:
        - **1.000**: No overlap, no adjustment needed
        - **0.700**: Moderate overlap, typical for financial data
        - **0.300**: Heavy overlap, very conservative adjustment
        - **0.126**: Extreme overlap, maximally conservative testing
        
        **Lower adjustment factors** = **stronger correlation** = **more conservative testing**
        
        ### **Confidence Intervals**
        - **95% range** of plausible iota values accounting for uncertainty
        - **Narrow intervals**: High precision, confident in the estimate
        - **Wide intervals**: High uncertainty, need more data or longer periods
        - **Intervals crossing zero**: Performance difference might not be meaningful
        
        ## Sample Reliability Assessment
        
        | Min Sample Size | Reliability | Interpretation |
        |-----------------|-------------|----------------|
        | **≥378 days** | HIGH_CONFIDENCE | ~1.5 years - excellent statistical power |
        | **≥189 days** | MODERATE_CONFIDENCE | ~9 months - reasonable statistical power |
        | **≥90 days** | LOW_CONFIDENCE | ~4.5 months - limited but usable |
        | **<90 days** | INSUFFICIENT_DATA | <90 days - insufficient for reliable statistics |
        
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
        """)
    
    with help_tab4:
        st.subheader("📈 Distribution Analysis Guide")
        
        st.markdown("""
        ## What is Distribution Analysis?
        
        Distribution analysis visualizes the historical in-sample performance distributions for each metric, with your out-of-sample values clearly marked for easy comparison.
        
        ### 🎯 Purpose
        - **Visual Comparison**: See exactly where your OOS performance falls relative to historical patterns
        - **Distribution Shape**: Understand the range and variability of historical performance
        - **Intuitive Interpretation**: Left of distribution = underperforming, right = outperforming
        - **Multi-Metric View**: Compare all four metrics simultaneously
        
        ## Understanding the Distribution Charts
        
        ### 📊 Chart Elements
        
        **Histogram Bars**: Show the frequency of different performance levels during the backtest period
        - **Height**: How often that performance level occurred
        - **Width**: Performance range for that bin
        - **Color**: Each metric has its own color for easy identification
        
        **Red Dashed Line**: Your out-of-sample performance value
        - **Position**: Shows exactly where your OOS performance falls
        - **Relative to distribution**: Left = underperforming, right = outperforming
        
        **Blue Dashed Line**: In-sample median (expected performance)
        - **Reference point**: Shows the "typical" performance level
        - **Comparison**: How far your OOS value is from the median
        
        ### 📈 Metric-Specific Interpretations
        
        **Annualized Return**:
        - **Left of distribution**: OOS returns below historical expectations
        - **Right of distribution**: OOS returns above historical expectations
        - **Units**: Percentage points (e.g., 15.2% = 15.2 percentage points)
        
        **Sharpe Ratio**:
        - **Left of distribution**: OOS risk-adjusted returns below historical expectations
        - **Right of distribution**: OOS risk-adjusted returns above historical expectations
        - **Units**: Risk-adjusted return ratio (dimensionless)
        
        **Cumulative Return**:
        - **Left of distribution**: OOS total returns below historical expectations
        - **Right of distribution**: OOS total returns above historical expectations
        - **Units**: Percentage points (e.g., 25.8% = 25.8 percentage points)
        
        **Sortino Ratio**:
        - **Left of distribution**: OOS downside risk management below historical expectations
        - **Right of distribution**: OOS downside risk management above historical expectations
        - **Units**: Downside risk-adjusted return ratio (dimensionless)
        
        ## Interpreting Distribution Results
        
        ### ✅ Healthy Patterns
        - **OOS values near median**: Performance matches historical expectations
        - **OOS values within distribution**: Performance is within normal historical range
        - **Consistent across metrics**: All metrics show similar relative performance
        
        ### ⚠️ Concerning Patterns
        - **OOS values far left of distribution**: Significant underperformance
        - **OOS values outside distribution**: Performance outside historical range
        - **Inconsistent across metrics**: Some metrics performing well, others poorly
        
        ### 🚨 Critical Patterns
        - **OOS values at extreme left**: Severe underperformance
        - **Multiple metrics showing poor performance**: Systematic issues
        - **OOS values well outside distribution**: Unusual performance requiring investigation
        
        ## Using Distribution Analysis
        
        ### 🎯 Quick Assessment
        1. **Look at red lines**: Where do they fall relative to the histograms?
        2. **Compare to blue lines**: How far from the median?
        3. **Check consistency**: Are all metrics telling the same story?
        4. **Consider magnitude**: How far outside the distribution?
        
        ### 📊 Detailed Analysis
        - **Distribution shape**: Wide distributions suggest high variability
        - **Skewness**: Asymmetric distributions indicate bias
        - **Outliers**: Extreme values in historical data
        - **OOS position**: Percentile rank within historical distribution
        
        ## Best Practices
        
        ### ✅ What to Look For
        - **Consistent positioning**: All metrics showing similar relative performance
        - **Reasonable distance**: OOS values not too far from median
        - **Distribution coverage**: OOS values within historical range
        
        ### ⚠️ Warning Signs
        - **Extreme positioning**: OOS values at distribution edges
        - **Inconsistent patterns**: Different metrics showing opposite results
        - **Outside distribution**: OOS values beyond historical range
        
        ### 🔍 Follow-up Actions
        - **If concerning**: Run rolling analysis for temporal patterns
        - **If critical**: Review strategy parameters and market conditions
        - **If unusual**: Investigate specific time periods or market events
        """)
    
    with help_tab5:
        st.subheader("🔄 Rolling Window Analysis Guide")
        
        st.markdown("""
        ## What is Rolling Window Analysis?
        
        Rolling window analysis divides your out-of-sample period into multiple overlapping time windows to detect **overfitting patterns** and **performance degradation** over time using sophisticated metrics.
        
        ### 🎯 Purpose
        - **Advanced Overfitting Detection**: Comprehensive analysis of performance patterns over time
        - **Sophisticated Trend Analysis**: Time spent above/below zero, magnitude analysis, area integration
        - **Early Warning**: Catch degradation before it becomes severe
        - **Strategy Validation**: Confirm consistent performance vs. lucky periods
        
        ## How It Works
        
        ### Step 1: Window Creation
        - Your OOS period is divided into overlapping windows (e.g., 6-month windows)
        - Windows advance by smaller steps (e.g., 1-month steps)
        - Creates multiple "mini out-of-sample" periods for analysis
        
        ### Step 2: Historical Comparison
        - Each window is compared against the same IS slice distribution
        - Iota calculated for each window vs. historical expectations
        - Tracks how each metric performs over time
        
        ### Step 3: Performance Tracking
        - Iota values calculated for each time window
        - Tracks how performance compares to backtest expectations over time
        - Multiple metrics analyzed independently
        
        ### Step 4: Advanced Risk Assessment
        - **Sophisticated Scoring**: Time spent above/below zero, magnitude analysis, area integration, consistency
        - **Risk Classification**: MINIMAL → LOW → MODERATE → HIGH → CRITICAL
        - **Comprehensive Analysis**: Multiple factors including area imbalance and performance variance
        
        ## Interpreting Rolling Analysis Results
        
        ### 🎯 Decay Risk Levels
        
        | Risk Level | Degradation Score | Interpretation | Action Required |
        |------------|-------------------|----------------|-----------------|
        | **MINIMAL** | 0-1 | ✅ Consistent performance | Continue monitoring |
        | **LOW** | 2-4 | ℹ️ Minor inconsistencies | Periodic review |
        | **MODERATE** | 5-7 | ⚠️ Some degradation detected | Monitor closely |
        | **HIGH** | 8-11 | 🚨 Significant degradation | Consider re-assessment |
        | **CRITICAL** | 12+ | 💀 Severe degradation | Likely overfit, urgent review |
        
        ### 📈 Understanding the Rolling Plot
        
        **Key Elements:**
        - **Gray line at ι = 0**: Neutral performance (matches historical median)
        - **Green dotted line at ι = +0.5**: Good performance threshold
        - **Red dotted line at ι = -0.5**: Poor performance threshold
        - **Colored lines**: Individual metrics (Sharpe, Cumulative Return, Sortino)
        - **Smoothing**: 3-period moving average reduces noise
        
        **Healthy Patterns (Low Risk):**
        - ✅ Iotas fluctuate around zero with minimal time below -0.5
        - ✅ Multiple metrics show similar, stable patterns
        - ✅ Less than 20% of time spent underperforming
        
        **Warning Patterns (Moderate Risk):**
        - ⚠️ Some periods of underperformance (20-40% of time below -0.5)
        - ⚠️ Some metrics underperforming while others stable
        - ⚠️ Moderate proportion of time below performance threshold
        
        **Critical Patterns (High Risk):**
        - 🚨 Extended periods of underperformance (40-60% of time below -0.5)
        - 🚨 Multiple metrics consistently underperforming
        - 🚨 High proportion of time below -0.5 threshold
        - 🚨 Wide divergence between different metrics
        
        ### 🔍 Metric-Specific Performance
        
        **Individual metric underperformance indicates:**
        - **Sharpe Ratio under -0.5**: Risk-adjusted performance below backtest expectations
        - **Cumulative Return under -0.5**: Total returns falling behind historical performance
        - **Sortino Ratio under -0.5**: Downside risk management below expected levels
        
        ## Advanced Decay Risk Assessment Components
        
        The sophisticated risk assessment considers multiple factors:
        
        ### 📊 Time-Based Analysis
        - **Time below zero**: Percentage of periods with negative iota values
        - **Time above zero**: Percentage of periods with positive iota values
        - **Scoring**: >80% below zero (+4), >60% (+3), >40% (+2), >20% (+1)
        
        ### 📉 Magnitude Analysis
        - **Average negative magnitude**: Mean of all negative iota values
        - **Average positive magnitude**: Mean of all positive iota values
        - **Scoring**: Avg negative < -1.5 (+4), < -1.0 (+3), < -0.5 (+2), < -0.2 (+1)
        
        ### 📈 Area Integration
        - **Positive area**: Sum of all positive iota values
        - **Negative area**: Sum of all negative iota values
        - **Area ratio**: Ratio of negative to positive areas
        - **Scoring**: Area ratio > 3.0 (+3), > 2.0 (+2), > 1.5 (+1)
        
        ### 🔄 Consistency Analysis
        - **Performance variance**: Standard deviation of negative performance
        - **Scoring**: High variance in negative performance (+1-2 points)
        
        ## Actionable Insights
        
        ### ✅ If Rolling Analysis Shows Low Risk:
        - **Continue current strategy** with confidence
        - **Consider scaling** position sizes if conservative
        - **Periodic monitoring** (monthly/quarterly reviews)
        - **Document current parameters** for future reference
        - **Shorter OOS periods should be taken with a grain of salt
        
        ### ⚠️ If Rolling Analysis Shows Moderate Risk:
        - **Increase monitoring frequency** (weekly reviews)
        - **Review recent market conditions** for regime changes
        - **Consider minor parameter adjustments** if trend continues
        - **Prepare contingency plans** for further degradation
        - **Shorter OOS periods should be taken with a grain of salt
        
        ### 🚨 If Rolling Analysis Shows High/Critical Decay Risk:
        - **Urgent strategy review** required
        - **Reduce position sizes** immediately
        - **Extended backtesting** with longer historical periods
        - **Parameter re-optimization** or strategy replacement
        - **Daily monitoring** until stabilization
        - **Shorter OOS periods should be taken with a grain of salt
        
        ## Technical Parameters
        
        ### 🔧 Window Sizing (Adaptive)
        - **2+ years OOS**: 126-day windows (6 months)
        - **1-2 years OOS**: 84-day windows (4 months)
        - **9+ months OOS**: 63-day windows (3 months)
        - **3-9 months OOS**: Adaptive sizing (minimum 21 days)
        
        ### 📏 Step Sizing
        - **Default**: Window size ÷ 8 (e.g., 126÷8 = ~16 days)
        - **Minimum**: 5 days
        - **Purpose**: Balance between granularity and computational efficiency
        
        ### 🎯 Minimum Requirements
        - **Minimum windows**: 6 for meaningful trend analysis
        - **Maximum windows**: 60 (performance limitation)
        - **Minimum OOS period**: 90 days for any rolling analysis
        """)
    
    with help_tab6:
        st.subheader("❓ Frequently Asked Questions")
        
        st.markdown("""
        ## General Questions
        
        ### Q: What makes this different from just looking at returns?
        **A:** This tool provides **statistical context**. Getting 30% returns is great, but if your strategy had a backtest suggesting 300%, that's not good. However, if the backtest consistently got 20-40%, then 30% means your strategy doesn't look overfit. That's excellent (and honestly, pretty rare)!
        
        ### Q: Can this predict future performance?
        **A:** **No.** This is a **retrospective analysis tool**. It tells you how unusual your recent performance has been relative to history, but cannot predict what will happen next.
        
        ### Q: Why do I need both core analysis AND rolling analysis?
        **A:** 
        - **Core analysis**: Overall assessment of your entire OOS period
        - **Distribution analysis**: Visualize in-sample distributions with OOS values marked
        - **Rolling analysis**: Detects **when** and **how** performance changes over time
        - **Together**: Complete picture of strategy health, decay and overfitting risk
        
        ## Interpretation Questions
        
        ### Q: What's a "good" iota score?
        **A:** 
        - **ι > +0.5**: Outstanding outperformance
        - **ι ≈ 0**: Performing as expected (this is actually good!)
        - **ι < -0.5**: Concerning underperformance
        - **Remember**: ι = 0 means your strategy is working exactly as the backtest suggested
        
        ### Q: Why are some metrics significant and others not?
        **A:** Different metrics measure different aspects of performance:
        - **Sharpe Ratio**: Risk-adjusted returns
        - **Cumulative Return**: Total returns
        - **Sortino Ratio**: Downside risk management
        - Your strategy might excel in one area but not others
        
        ### Q: What does "autocorrelation adjusted" mean?
        **A:** When using overlapping slices, adjacent periods share most of their data, violating statistical independence assumptions. The adjustment makes p-values more conservative (harder to achieve significance) to account for this correlation.
        
        ## Technical Questions
        
        ### Q: How many IS slices should I use?
        **A:**
        - **50-100**: Good balance of statistics and speed
        - **100-200**: Better statistics, slower computation
        - **More isn't always better**: Diminishing returns beyond ~200 slices
        
        ### Q: Should I use overlapping or non-overlapping slices?
        **A:**
        - **Overlapping (recommended)**: More data, better statistics, uses autocorrelation adjustment
        - **Non-overlapping**: Simpler statistics, less data, faster computation
        
        ### Q: How long should my OOS period be?
        **A:**
        - **Minimum**: 3 months (90 days)
        - **Recommended**: 6-12 months for meaningful analysis
        - **Ideal**: 1-2 years for high confidence
        - **Longer isn't always better**: Very long periods may include regime changes
        
        ## Common Issues
        
        ### Q: "Insufficient data" error - what do I do?
        **A:** 
        - Extend your data start date further back
        - Reduce the number of IS slices
        - Ensure your OOS period is at least 90 days
        - Check that your symphony has enough historical data
        
        ### Q: Analysis is very slow - how to speed up?
        **A:**
        - Reduce number of IS slices (try 50 instead of 100)
        - Use non-overlapping slices
        - Disable rolling analysis for faster core analysis only
        - Use shorter date ranges for initial testing
        
        ### Q: All my iotas are near zero - is this bad?
        **A:** **No!** Iotas near zero mean your strategy is performing **exactly as expected** based on historical patterns. This is actually **good** - it means your backtest was realistic and your strategy is working as designed.
        
        ### Q: Rolling analysis shows high decay risk - what now?
        **A:**
        1. **Don't panic** - check if it's due to recent market conditions
        2. **Review strategy parameters** - may need adjustment for current market
        3. **Extend backtesting period** - include more market regimes
        4. **Consider position size reduction** - while investigating
        5. **Monitor** - track if degradation continues or stabilizes
        
        ## Best Practices
        
        
        ### Q: What exclusion periods should I use?
        **A:** Consider excluding:
        - **Market crashes** (2020 COVID crash, 2008 financial crisis)
        - **Extreme volatility periods** that aren't representative
        - **Data quality issues** (corporate actions, splits, etc.)
        - **Be conservative**: Only exclude truly exceptional periods
        
        ### Q: My strategy looks good in core analysis but bad in rolling analysis - which to trust?
        **A:** 
        - **Both are important** - they tell different stories
        - **Core analysis**: Overall performance vs. expectations
        - **Rolling analysis**: Performance consistency over time
        - **Action**: Monitor closely, may indicate strategy evolution needed
        
        ## Troubleshooting
        
        ### Q: P-values show as 0.000 - is this an error?
        **A:** No, this typically means p < 0.001 (very statistically significant). The display rounds to 3 decimal places, so very small p-values appear as 0.000.
        
        ### Q: Confidence intervals are very wide - what does this mean?
        **A:** 
        - **High uncertainty** in the iota estimate
        - **Need more data** - longer OOS period or more IS slices
        - **Results less reliable** - interpret with caution
        - **Consider extending** analysis period
        
        ### Q: Sortino ratio shows "∞ (no downside)" - is this normal?
        **A:** Yes! This means your strategy had **no negative return days** during that period. The Sortino ratio becomes infinite when there's no downside volatility to measure.
        
        ## Data Quality
        
        ### Q: How do I know if my Composer data is good?
        **A:**
        - Check for **missing dates** (gaps in daily returns)
        - Look for **extreme outliers** (>±50% daily returns)
        - Verify **corporate actions** are handled correctly
        - Compare with **external data sources** if possible
        
        ### Q: My symphony stopped working mid-analysis - what happened?
        **A:**
        - **Composer API limits**: Try again in a few minutes
        - **Symphony privacy settings**: Ensure symphony is public or accessible
        - **Network issues**: Check internet connection
        - **Data corruption**: Try with a different date range
        
        ## Contact & Support
        
        ### Q: I found a bug or have a feature request - where do I report it?
        **A:** Reach out to @gobi on discord and I'd be happy to chat.
        
        ### Q: Can I use this for non-Composer strategies?
        **A:** The tool is designed for Composer symphonies, but the statistical methodology can be adapted for any return series with appropriate modifications to the data input module.
        """)
    
    # Add footer with version info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
        <p><strong>Iota Calculator - Core and Rolling Analysis</strong></p>
        <p>Edge persistence, overfitness, and decay risk assessment tool</p>
        <p>Questions? Reach out to @gobi on Discord</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
