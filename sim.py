# Monte Carlo simulations of forward returns using composer back tests
# update 1: fixed drawdown duration to date mapping, added rolling walk mode: 20250418
# update 2: updated price fetching to current yfinance package: 20250507
# update 3: added modes Expanding window, Forward Forecast, added option to use returns from a previous run, menu looping
# prairie@Investor's Collaborative 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta, date
import os
from matplotlib.gridspec import GridSpec
import requests
from typing import List, Dict

# Set random seed for reproducibility
np.random.seed(42)

def convert_trading_date(date_int):
    """
    Convert trading date integer to datetime object
    """
    date_1 = datetime.strptime("01/01/1970", "%m/%d/%Y")
    dt = date_1 + timedelta(days=int(date_int))
    return dt


class YahooFinanceAPI:
    """Fetches historical price data using the yfinance package."""

    def __init__(self, session=None):
        """
        Initialize the Yahoo Finance API client.

        Args:
            session: Optional requests session (not used with yfinance but kept for compatibility)
        """
        # Try importing yfinance
        try:
            import yfinance as yf
            self.yf = yf
            print("Successfully initialized yfinance package")
        except ImportError:
            print("yfinance package is not installed. Please install it with: pip install yfinance")
            raise ImportError("yfinance package is required to use the Yahoo Finance API")

        # Dictionary to map special tickers to their Yahoo Finance format
        self.ticker_map = {
            'BRK/B': 'BRK-B'
        }

        # Default settings
        self.rate_limit_delay = 1.0  # seconds between requests
        self.use_batch_download = True  # Set to True to use batch downloading instead of individual downloads
        self.batch_size = 5  # Number of symbols per batch when using batch download

    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Fetch historical price data for multiple symbols.

        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping symbols to price series
        """

        print(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")

        # Map symbols to Yahoo Finance format if needed
        mapped_symbols = {}
        for symbol in symbols:
            yahoo_symbol = self.ticker_map.get(symbol, symbol)
            if yahoo_symbol != symbol:
                print(f"Special ticker handling: Mapping {symbol} to {yahoo_symbol} for Yahoo Finance")
            mapped_symbols[yahoo_symbol] = symbol

        # Choose download method
        if self.use_batch_download and len(symbols) > 1:
            print(f"Using batch download method for {len(symbols)} symbols")
            return self._batch_download(mapped_symbols, start_date, end_date)
        else:
            print(f"Using individual download method for {len(symbols)} symbols")
            return self._individual_download(mapped_symbols, start_date, end_date)

    def _individual_download(self, mapped_symbols: Dict[str, str], start_date: str, end_date: str) -> Dict[
        str, pd.Series]:
        """
        Download data for each symbol individually (more reliable but slower).

        Args:
            mapped_symbols: Dictionary mapping Yahoo symbols to original symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping original symbols to price series
        """
        import time
        import numpy as np

        price_data = {}

        # Process symbols one at a time
        for yahoo_symbol, original_symbol in mapped_symbols.items():
            print(f"Fetching data for {original_symbol}")

            # Try with the primary method
            try:
                # Create a ticker object
                ticker_obj = self.yf.Ticker(yahoo_symbol)

                # Get historical data
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True  # Use adjusted data
                )

                if data.empty:
                    print(f"No data returned for {original_symbol}")
                    continue

                # Extract the 'Close' prices
                if 'Close' in data.columns:
                    series = data['Close'].copy()

                    # Clean the data
                    series = series.dropna()
                    series = series.astype(np.float32)

                    # Make sure the index is timezone naive
                    if series.index.tz is not None:
                        series.index = series.index.tz_convert('America/New_York')
                        series.index = series.index.tz_localize(None)

                    # Remove duplicates
                    series = series[~series.index.duplicated(keep='last')]

                    if not series.empty:
                        series.name = original_symbol  # Use the original symbol name
                        price_data[original_symbol] = series
                        print(
                            f"Successfully retrieved {original_symbol}: {len(series)} points "
                            f"from {series.index[0].strftime('%Y-%m-%d')} "
                            f"to {series.index[-1].strftime('%Y-%m-%d')}"
                        )
                    else:
                        print(f"Series was empty after cleaning for {original_symbol}")
                else:
                    print(f"No Close column found for {original_symbol}")

            except Exception as e:
                print(f"Error fetching data for {original_symbol}: {str(e)}")
                # Try with a modified symbol (some ETFs need this)
                if '-' not in yahoo_symbol and '/' not in yahoo_symbol:
                    mod_symbol = f"{yahoo_symbol}-USD"
                    print(f"Trying alternative symbol format: {mod_symbol}")

                    try:
                        # Create a ticker object with the modified symbol
                        ticker_obj = self.yf.Ticker(mod_symbol)

                        # Get historical data
                        data = ticker_obj.history(
                            start=start_date,
                            end=end_date,
                            auto_adjust=True
                        )

                        if not data.empty and 'Close' in data.columns:
                            series = data['Close'].copy()
                            series = series.dropna()
                            series = series.astype(np.float32)

                            # Make sure the index is timezone naive
                            if series.index.tz is not None:
                                series.index = series.index.tz_convert('America/New_York')
                                series.index = series.index.tz_localize(None)

                            # Remove duplicates
                            series = series[~series.index.duplicated(keep='last')]

                            if not series.empty:
                                series.name = original_symbol  # Use the original symbol name
                                price_data[original_symbol] = series
                                print(
                                    f"Successfully retrieved {original_symbol} (as {mod_symbol}): {len(series)} points "
                                    f"from {series.index[0].strftime('%Y-%m-%d')} "
                                    f"to {series.index[-1].strftime('%Y-%m-%d')}"
                                )
                    except Exception as alt_e:
                        print(f"Error with alternative symbol format for {original_symbol}: {str(alt_e)}")

            # Add a small delay between requests to avoid rate limiting
            time.sleep(self.rate_limit_delay)

        return price_data

    def _batch_download(self, mapped_symbols: Dict[str, str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Download data for multiple symbols in batches (faster but less reliable).

        Args:
            mapped_symbols: Dictionary mapping Yahoo symbols to original symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping original symbols to price series
        """
        import time
        import pandas as pd
        import numpy as np

        price_data = {}
        yahoo_symbols = list(mapped_symbols.keys())

        # Process symbols in batches
        for i in range(0, len(yahoo_symbols), self.batch_size):
            batch = yahoo_symbols[i:i + self.batch_size]
            print(f"Fetching batch of {len(batch)} symbols (batch {i // self.batch_size + 1})")

            try:
                # Download data for the entire batch
                data = self.yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,  # Use adjusted data
                    actions=False,  # Don't include dividends
                    progress=False  # Don't display progress bar
                )

                # Process batch results
                if len(batch) == 1:
                    # Special case for single ticker (different data structure)
                    yahoo_symbol = batch[0]
                    original_symbol = mapped_symbols[yahoo_symbol]

                    if 'Close' in data.columns:
                        series = data['Close'].copy()

                        # Clean the data
                        series = series.dropna()
                        series = series.astype(np.float32)

                        # Make sure the index is timezone naive
                        if series.index.tz is not None:
                            series.index = series.index.tz_convert('America/New_York')
                            series.index = series.index.tz_localize(None)

                        # Remove duplicates
                        series = series[~series.index.duplicated(keep='last')]

                        if not series.empty:
                            series.name = original_symbol
                            price_data[original_symbol] = series
                            print(f"Successfully retrieved {original_symbol}: {len(series)} points")

                else:
                    # Process multiple tickers
                    for yahoo_symbol in batch:
                        original_symbol = mapped_symbols[yahoo_symbol]

                        try:
                            if yahoo_symbol in data.columns and 'Close' in data[yahoo_symbol].columns:
                                series = data[yahoo_symbol]['Close'].copy()

                                # Clean the data
                                series = series.dropna()
                                series = series.astype(np.float32)

                                # Make sure the index is timezone naive
                                if series.index.tz is not None:
                                    series.index = series.index.tz_convert('America/New_York')
                                    series.index = series.index.tz_localize(None)

                                # Remove duplicates
                                series = series[~series.index.duplicated(keep='last')]

                                if not series.empty:
                                    series.name = original_symbol
                                    price_data[original_symbol] = series
                                    print(f"Successfully retrieved {original_symbol}: {len(series)} points")
                                else:
                                    print(f"Empty data for {original_symbol} after cleaning")
                            else:
                                print(f"No data found for {original_symbol} in batch results")
                        except Exception as e:
                            print(f"Error processing batch data for {original_symbol}: {str(e)}")

            except Exception as e:
                print(f"Error fetching batch {i // self.batch_size + 1}: {str(e)}")

            # Add delay between batches
            if i + self.batch_size < len(yahoo_symbols):
                time.sleep(self.rate_limit_delay * 2)  # Longer delay between batches

        # Check for failed tickers and retry individually
        failed_symbols = {s: mapped_symbols[s] for s in mapped_symbols if mapped_symbols[s] not in price_data}

        if failed_symbols:
            print(f"Retrying {len(failed_symbols)} failed symbols individually...")
            retry_results = self._individual_download(failed_symbols, start_date, end_date)
            price_data.update(retry_results)

        return price_data


def fetch_backtest(id, start_date, end_date):
    """
    Fetch backtest data from Composer API
    """
    if id.endswith('/details'):
        id = id.split('/')[-2]
    else:
        id = id.split('/')[-1]

    payload = {
        "capital": 100000,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": 0.0005,
        "start_date": start_date,
        "end_date": end_date,
    }

    url = f"https://backtest-api.composer.trade/api/v2/public/symphonies/{id}/backtest"

    data = requests.post(url, json=payload)
    jsond = data.json()
    symphony_name = jsond['legend'][id]['name']

    holdings = jsond["last_market_days_holdings"]

    tickers = []
    for ticker in holdings:
        tickers.append(ticker)

    # Extract allocations
    allocations = jsond["tdvm_weights"]
    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(0.0, index=date_range, columns=tickers)

    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            percent = allocations[ticker][date_int]
            df.at[trading_date, ticker] = percent

    return df, symphony_name, tickers

def calculate_portfolio_returns(allocations_df, tickers):
    """
    Calculate daily portfolio returns with properly normalized dates using allocation weighting
    and correct compounding.
    """
    # Find the first row with at least one non-zero value
    first_valid_index = allocations_df[(abs(allocations_df) > 0.000001).any(axis=1)].first_valid_index()

    # Get rid of data prior to start of backtest and non-trading days
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0

    # Add $USD column if not present
    if '$USD' not in allocations_df.columns:
        allocations_df['$USD'] = 0

    # IMPORTANT: Normalize allocation dates to remove time component
    # Convert to datetime and keep only the date part, then convert back to datetime
    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()

    # Extract unique tickers
    unique_tickers = {ticker for ticker in tickers if ticker != '$USD'}

    # Fetch historical prices with adequate buffer
    start_date = allocations_df.index.min() - timedelta(days=10)
    end_date = allocations_df.index.max() + timedelta(days=10)

    print(f"Fetching price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Initialize Yahoo Finance API
    yahoo_api = YahooFinanceAPI()

    # Fetch historical prices
    prices_data = yahoo_api.fetch_historical_data(
        list(unique_tickers),
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    # Create price DataFrame
    prices = pd.DataFrame({ticker: prices_data[ticker] for ticker in prices_data})

    # IMPORTANT: Normalize price dates to remove time component
    prices.index = pd.to_datetime(prices.index).normalize()

    # Add $USD column with value 1.0
    prices['$USD'] = 1.0

    # Make sure we have all the tickers
    for ticker in tickers:
        if ticker not in prices.columns and ticker != '$USD':
            print(f"Warning: Price data for {ticker} not found. Setting to NaN.")
            prices[ticker] = np.nan

    # Forward fill missing values
    prices = prices.ffill()
    prices = prices.bfill()
    prices = prices.fillna(1.0)

    # Reorder columns to match tickers
    prices = prices[tickers]

    # Sort DataFrames by index
    allocations_df.sort_index(inplace=True)
    prices.sort_index(inplace=True)

    # Print date info with normalized dates
    price_dates = sorted(prices.index)
    alloc_dates = sorted(allocations_df.index)
    print(f"Retrieved {len(price_dates)} price dates from {price_dates[0].date()} to {price_dates[-1].date()}")
    print(f"Have {len(alloc_dates)} allocation dates from {alloc_dates[0].date()} to {alloc_dates[-1].date()}")

    # Check if all allocation dates exist in price dates
    missing_dates = set(alloc_dates) - set(price_dates)
    if missing_dates:
        print(f"Found {len(missing_dates)} allocation dates without exact price date matches")
        print(f"First few missing dates: {[d.date() for d in sorted(missing_dates)[:5]]}")
    else:
        print("All allocation dates have exact matching price dates!")

    # Create a simple dictionary to store price data by date
    price_dict = {}
    for date, row in prices.iterrows():
        date_key = date.strftime('%Y-%m-%d')  # Use string as key for consistent matching
        price_dict[date_key] = row

    # Create dictionaries to store price changes by ticker and date
    price_changes = {}

    # Calculate daily price changes for each ticker
    for ticker in tickers:
        if ticker == '$USD':
            continue  # Skip cash

        price_changes[ticker] = {}
        ticker_prices = prices[ticker]

        # Calculate daily percentage changes
        for i in range(1, len(price_dates)):
            today = price_dates[i]
            yesterday = price_dates[i - 1]

            today_price = ticker_prices.loc[today]
            yesterday_price = ticker_prices.loc[yesterday]

            # Calculate daily percentage change
            if yesterday_price is not None:
                daily_change = ((today_price / yesterday_price) - 1) * 100
                price_changes[ticker][today.strftime('%Y-%m-%d')] = daily_change

    # Initialize daily returns
    daily_returns = pd.Series(index=allocations_df.index[1:], dtype=float)

    # Calculate portfolio returns using the weighted allocation approach
    for i in range(1, len(allocations_df)):
        today_date = allocations_df.index[i]

        today_key = today_date.strftime('%Y-%m-%d')

        # Get yesterday's allocations (these are the active allocations for calculating today's return)
        allocations_yday = allocations_df.iloc[i - 1, :] / 100.0  # Convert to 0-1 range

        # Calculate weighted return for the day
        portfolio_daily_return = 0.0

        for ticker in tickers:
            if ticker == '$USD':
                # Cash has 0% return
                continue

            ticker_allocation = allocations_yday[ticker]

            if ticker_allocation > 0:
                if today_key in price_changes.get(ticker, {}):
                    # Apply allocation weighting to the ticker's return
                    ticker_return = price_changes[ticker][today_key]
                    portfolio_daily_return += ticker_allocation * ticker_return

        # Store the daily return
        daily_returns.iloc[i - 1] = portfolio_daily_return

        # Log information for the last few days
        if i >= len(allocations_df) - 5:
            print(f"\nCalculating return for {today_date.date()}:")
            print(f"Portfolio daily return: {portfolio_daily_return:.4f}%")
            print("Ticker contributions:")

            for ticker in tickers:
                if ticker == '$USD' or allocations_yday[ticker] <= 0:
                    continue

                if today_key in price_changes.get(ticker, {}):
                    ticker_return = price_changes[ticker][today_key]
                    contribution = allocations_yday[ticker] * ticker_return
                    print(f"  {ticker}: {allocations_yday[ticker] * 100:.2f}% allocation, "
                          f"{ticker_return:.4f}% return, {contribution:.4f}% contribution")

    # Print return statistics
    print("\n--- RETURN STATISTICS ---")
    print(f"Average daily return: {daily_returns.mean():.4f}%")
    print(f"Min/Max daily return: {daily_returns.min():.4f}% / {daily_returns.max():.4f}%")
    print(f"Positive days: {(daily_returns > 0).sum()} ({(daily_returns > 0).mean() * 100:.2f}%)")

    # Ensure length alignment
    if len(daily_returns) != len(allocations_df.index[1:]):
        print(
            f"Warning: Return length ({len(daily_returns)}) doesn't match allocation dates ({len(allocations_df.index[1:])})")
        # Trim to ensure match
        min_len = min(len(daily_returns), len(allocations_df.index[1:]))
        daily_returns = daily_returns[:min_len]
        dates = allocations_df.index[:min_len + 1]  # +1 because first date has no return
    else:
        dates = allocations_df.index

    return daily_returns, dates

def run_monte_carlo_simulation(returns, num_simulations=10000, simulation_length=None, annual_periods=252):
    """
    Run Monte Carlo simulation using separate sampling for positive and negative returns
    """
    if simulation_length is None:
        simulation_length = len(returns)

    returns_array = np.array(returns)

    # Separate positive and negative returns
    positive_returns = returns_array[returns_array > 0]
    negative_returns = returns_array[returns_array <= 0]

    # Calculate probabilities
    prob_positive = len(positive_returns) / len(returns_array)

    print(f"Return characteristics - Probability of positive return: {prob_positive:.4f}")
    print(f"Positive returns - Mean: {np.mean(positive_returns):.4f}, Std: {np.std(positive_returns):.4f}")
    print(f"Negative returns - Mean: {np.mean(negative_returns):.4f}, Std: {np.std(negative_returns):.4f}")

    # Initialize arrays to store simulation results
    cumulative_returns = np.zeros((num_simulations, simulation_length + 1))
    cumulative_returns[:, 0] = 0  # Start with 0% return

    # Arrays to store Sharpe ratios and max drawdowns
    sharpe_ratios = np.zeros(num_simulations)
    max_drawdowns = np.zeros(num_simulations)

    # Arrays to store drawdown durations
    max_drawdown_durations = np.zeros(num_simulations)
    total_drawdown_days = np.zeros(num_simulations)

    # Run simulations
    for i in range(num_simulations):
        # Generate random returns by sampling separately from positive and negative returns
        simulated_returns = np.zeros(simulation_length)
        for j in range(simulation_length):
            if np.random.random() < prob_positive:
                # Sample from positive returns
                simulated_returns[j] = np.random.choice(positive_returns)
            else:
                # Sample from negative returns
                simulated_returns[j] = np.random.choice(negative_returns)

        # Calculate cumulative returns
        cum_return = 0
        cum_returns = [cum_return]
        peak = 0
        max_drawdown = 0

        # Track drawdown durations
        in_drawdown = False
        current_drawdown_duration = 0
        max_dd_duration = 0
        total_dd_days = 0

        for r in simulated_returns:
            # Convert daily percentage return to decimal
            r_decimal = r / 100.0

            # Calculate new cumulative return (compounded)
            cum_return = (1 + cum_return / 100) * (1 + r_decimal) * 100 - 100
            cum_returns.append(cum_return)

            # Update peak and calculate drawdown
            if cum_return > peak:
                peak = cum_return
                # End of drawdown period
                if in_drawdown:
                    in_drawdown = False
                    max_dd_duration = max(max_dd_duration, current_drawdown_duration)
                    current_drawdown_duration = 0

            # Calculate drawdown as a percentage of peak value
            drawdown = ((peak - cum_return) / (1 + peak / 100)) if peak > 0 else 0

            # Track drawdown periods
            if drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                if in_drawdown:
                    current_drawdown_duration += 1
                    total_dd_days += 1

            max_drawdown = max(max_drawdown, drawdown)

        # If still in drawdown at end of series, update max_drawdown_duration
        if in_drawdown:
            max_dd_duration = max(max_dd_duration, current_drawdown_duration)

        cumulative_returns[i, :] = cum_returns

        # Calculate Sharpe ratio
        annual_return = cum_return * (annual_periods / simulation_length)
        annual_volatility = np.std(simulated_returns) * np.sqrt(annual_periods)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        sharpe_ratios[i] = sharpe_ratio
        max_drawdowns[i] = max_drawdown
        max_drawdown_durations[i] = max_dd_duration
        total_drawdown_days[i] = total_dd_days

    # Calculate percentiles for paths
    percentile_5 = np.percentile(cumulative_returns, 5, axis=0)
    percentile_25 = np.percentile(cumulative_returns, 25, axis=0)
    percentile_50 = np.percentile(cumulative_returns, 50, axis=0)
    percentile_75 = np.percentile(cumulative_returns, 75, axis=0)
    percentile_95 = np.percentile(cumulative_returns, 95, axis=0)

    # Calculate final returns for all simulations
    final_returns = cumulative_returns[:, -1]

    results = {
        'final_returns': final_returns,
        'paths': cumulative_returns,
        'percentiles': {
            '5': percentile_5,
            '25': percentile_25,
            '50': percentile_50,
            '75': percentile_75,
            '95': percentile_95
        },
        'sharpe_ratios': sharpe_ratios,
        'max_drawdowns': max_drawdowns,
        'max_drawdown_durations': max_drawdown_durations,
        'total_drawdown_days': total_drawdown_days
    }

    return results


def analyze_drawdowns(returns, output_dir, period_length, test_start_date, test_end_date, portfolio_name, dates=None):
    """
    Analyze drawdowns and create visualizations

    Parameters:
    -----------
    returns : list
        List of cumulative returns
    output_dir : str
        Directory to save output files
    period_length : int
        Length of the test period in days
    test_start_date : str
        Start date of the test period
    test_end_date : str
        End date of the test period
    portfolio_name : str
        Name of the portfolio for file naming
    dates : list, optional
        List of date strings corresponding to returns data
    """
    # Create a list of dates if not provided
    if dates is None:
        # Create synthetic dates starting from test_start_date
        start_date = pd.Timestamp(test_start_date)
        dates = [(start_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(returns))]

    # Ensure dates is the same length as returns
    if len(dates) != len(returns):
        print(f"Warning: Length mismatch between dates ({len(dates)}) and returns ({len(returns)})")
        # Use the shorter length
        min_length = min(len(dates), len(returns))
        dates = dates[:min_length]
        returns = returns[:min_length]

    # Convert dates to datetime objects for proper comparison
    date_objects = [pd.to_datetime(d).date() if isinstance(d, str) else d for d in dates]

    # Calculate drawdown periods and durations
    drawdown_periods = []
    drawdowns = []

    # Track the running peak
    running_peak = returns[0]
    current_drawdown_start_idx = None
    current_drawdown_start_value = None
    in_drawdown = False
    max_drawdown = 0
    max_drawdown_idx = 0

    # Loop through returns to calculate drawdowns and identify periods
    for i, value in enumerate(returns):
        # Update the running peak if we have a new high
        if value > running_peak:
            running_peak = value

            # If we were in a drawdown and now we've reached a new peak, the drawdown is over
            if in_drawdown:
                # Calculate the depth (percentage) of this drawdown period
                # IMPORTANT: Fix the decimal point issue - divide by 100 to get proper percentage
                min_value = min(returns[current_drawdown_start_idx:i])
                drawdown_depth = ((running_peak - min_value) / (1 + running_peak / 100))

                # Add this drawdown period to our list
                drawdown_periods.append({
                    'start_idx': current_drawdown_start_idx,
                    'end_idx': i,
                    'start_date': dates[current_drawdown_start_idx],
                    'end_date': dates[i],
                    'duration': i - current_drawdown_start_idx,
                    'max_drawdown': drawdown_depth,
                    # Calculate calendar days (not just trading days)
                    'calendar_days': (date_objects[i] - date_objects[current_drawdown_start_idx]).days
                })

                # Reset drawdown tracking
                in_drawdown = False
                current_drawdown_start_idx = None
                current_drawdown_start_value = None

        # Calculate current drawdown from peak - Fix decimal issue
        current_drawdown = ((running_peak - value) / (1 + running_peak / 100))
        drawdowns.append(current_drawdown)  # Convert to percentage

        # Update maximum drawdown if this is a new max
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_idx = i

        # Detect the start of a new drawdown
        if current_drawdown > 0 and not in_drawdown:
            in_drawdown = True
            current_drawdown_start_idx = i
            current_drawdown_start_value = running_peak  # Store the peak value, not the current value

    # If we're still in a drawdown at the end, add that period
    if in_drawdown:
        # Calculate the depth (percentage) of this final drawdown period
        min_value = min(returns[current_drawdown_start_idx:])
        drawdown_depth = ((running_peak - min_value) / (1 + running_peak / 100))

        # Add this drawdown period to our list
        drawdown_periods.append({
            'start_idx': current_drawdown_start_idx,
            'end_idx': len(returns) - 1,
            'start_date': dates[current_drawdown_start_idx],
            'end_date': dates[-1],
            'duration': len(returns) - current_drawdown_start_idx,
            'max_drawdown': drawdown_depth,  # Convert to percentage
            # Calculate calendar days (not just trading days)
            'calendar_days': (date_objects[-1] - date_objects[current_drawdown_start_idx]).days
        })

    # FIX: Recalculate max_drawdown from the drawdowns list to ensure consistency
    if drawdowns:
        max_drawdown = max(drawdowns)
        max_drawdown_idx = drawdowns.index(max_drawdown)

    # FIX: Recalculate drawdown periods to ensure they have the correct max_drawdown value
    for i, period in enumerate(drawdown_periods):
        start_idx = period['start_idx']
        end_idx = period['end_idx']
        period_drawdowns = drawdowns[start_idx:end_idx + 1]
        if period_drawdowns:
            period_max_dd = max(period_drawdowns)
            drawdown_periods[i]['max_drawdown'] = period_max_dd

    # FIX: Sort drawdown periods by max_drawdown (descending) for proper ranking
    drawdown_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)

    total_days_in_drawdown = sum(period['duration'] for period in drawdown_periods)
    # Use calendar days rather than trading days for significant periods
    significant_drawdown_days = sum(period['calendar_days'] for period in drawdown_periods
                                    if period['calendar_days'] > 20)

    # Find significant drawdown periods (duration > 20 calendar days)
    significant_periods = [p for p in drawdown_periods if p['calendar_days'] > 20]
    # Sort by max_drawdown (descending)
    significant_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)

    # Print top 5 significant drawdown periods with actual dates
    print("\nTop 5 Significant Drawdown Periods (>20 calendar days):")
    print(
        f"{'Rank':<5} {'Trading Days':<12} {'Calendar Days':<14} {'Max Drawdown':<15} {'Start Date':<12} {'End Date':<12}")
    print("-" * 70)

    for i, period in enumerate(significant_periods[:5], 1):
        print(
            f"{i:<5} {period['duration']:<12} {period['calendar_days']:<14} {period['max_drawdown']:.2f}%{' ':<9} {period['start_date']:<12} {period['end_date']:<12}")

    if not significant_periods:
        print("No significant drawdown periods (>20 calendar days) found.")

    # Calculate average statistics
    avg_drawdown_length = total_days_in_drawdown / len(drawdown_periods) if drawdown_periods else 0
    # Calculate average calendar days
    avg_calendar_days = sum(period['calendar_days'] for period in drawdown_periods) / len(
        drawdown_periods) if drawdown_periods else 0
    non_zero_drawdowns = [d for d in drawdowns if d > 0]
    avg_drawdown_depth = sum(non_zero_drawdowns) / len(non_zero_drawdowns) if non_zero_drawdowns else 0

    # Debug info
    print(f"\nDrawdown Calculation Debug:")
    print(f"Overall Max Drawdown: {max_drawdown:.2f}%")
    print(f"Number of drawdown periods found: {len(drawdown_periods)}")
    if drawdown_periods:
        max_period = max(drawdown_periods, key=lambda x: x['max_drawdown'])
        print(
            f"Largest period drawdown: {max_period['max_drawdown']:.2f}% (Trading Days: {max_period['duration']}, Calendar Days: {max_period['calendar_days']}, {max_period['start_date']} to {max_period['end_date']})")

    # Create the plot
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)

    # Create underwater plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
    ax1.plot(range(len(drawdowns)), drawdowns, color='red', linewidth=1)

    # Customize underwater plot
    ax1.set_title(f'Drawdown Over Time - {period_length} days')
    ax1.set_ylabel('Drawdown (%)')
    ax1.set_xlabel('Trading Days')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))

    # Add maximum drawdown line and annotation
    ax1.axhline(y=max_drawdown, color='darkred', linestyle='--', alpha=0.5)
    ax1.annotate(f'Maximum Drawdown: {max_drawdown:.2f}%\n'
                 f'Average Drawdown: {avg_drawdown_depth:.2f}%\n'
                 f'Avg Trading Days: {avg_drawdown_length:.1f}\n'
                 f'Avg Calendar Days: {avg_calendar_days:.1f}',
                 xy=(max_drawdown_idx, max_drawdown),
                 xytext=(10, 10),
                 textcoords='offset points',
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

    # Create cumulative returns plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(returns)), returns, color='blue', linewidth=1.5)

    # Mark drawdown periods
    for period in drawdown_periods:
        ax2.axvspan(period['start_idx'], period['end_idx'],
                    alpha=0.2, color='red',
                    label='_' * period['start_idx'])  # Unique labels to avoid duplicates in legend

    # Customize cumulative returns plot
    ax2.set_title(f'Cumulative Return with Drawdown Periods - {period_length} days')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_xlabel('Trading Days')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdown durations (middle row)
    ax3 = fig.add_subplot(gs[1, 0])

    durations = [period['duration'] for period in drawdown_periods]
    calendar_days = [period['calendar_days'] for period in drawdown_periods]
    max_drawdowns = [period['max_drawdown'] for period in drawdown_periods]

    if durations:
        # Create a grouped bar chart with both trading days and calendar days
        x = np.arange(len(durations))
        width = 0.35

        ax3.bar(x - width / 2, durations, width, color='blue', alpha=0.7, label='Trading Days')
        ax3.bar(x + width / 2, calendar_days, width, color='green', alpha=0.7, label='Calendar Days')

        # Add drawdown depth as text
        for i, (_, cal_days, dd) in enumerate(zip(durations, calendar_days, max_drawdowns)):
            ax3.text(i, cal_days + 1, f"{dd:.1f}%",
                     ha='center', va='bottom',
                     color='black', fontsize=8)

        # Customize drawdown durations plot
        ax3.set_title('Drawdown Duration by Episode')
        ax3.set_ylabel('Duration (Days)')
        ax3.set_xlabel('Drawdown Episode')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 1])

    if drawdowns:
        non_zero_drawdowns = [d for d in drawdowns if d > 0]
        if non_zero_drawdowns:
            sns.histplot(non_zero_drawdowns, bins=20, kde=True, ax=ax4, color='green')

            # Add vertical line for mean and maximum
            mean_dd = np.mean(non_zero_drawdowns)
            median_dd = np.median(non_zero_drawdowns)

            ax4.axvline(mean_dd, color='red', linestyle='--',
                        label=f'Mean: {mean_dd:.2f}%')
            ax4.axvline(median_dd, color='blue', linestyle='--',
                        label=f'Median: {median_dd:.2f}%')
            ax4.axvline(max_drawdown, color='black', linestyle='-',
                        label=f'Max: {max_drawdown:.2f}%')

            # Customize drawdown distribution plot
            ax4.set_title('Drawdown Magnitude Distribution')
            ax4.set_xlabel('Drawdown (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.7)
        else:
            ax4.text(0.5, 0.5, 'No non-zero drawdowns found',
                     ha='center', va='center', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No drawdowns found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown duration distribution (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])

    if durations:
        # Plot both trading days and calendar days as separate histograms
        sns.histplot(durations, bins=min(20, len(durations)), kde=True, ax=ax5, color='blue',
                     alpha=0.4, label='Trading Days')
        sns.histplot(calendar_days, bins=min(20, len(calendar_days)), kde=True, ax=ax5, color='green',
                     alpha=0.4, label='Calendar Days')

        # Add vertical line for mean and maximum of calendar days
        mean_duration = np.mean(durations)
        mean_calendar = np.mean(calendar_days)
        median_duration = np.median(durations)
        median_calendar = np.median(calendar_days)
        max_calendar = max(calendar_days)

        ax5.axvline(mean_calendar, color='darkgreen', linestyle='--',
                    label=f'Mean Calendar: {mean_calendar:.1f} days')
        ax5.axvline(median_calendar, color='green', linestyle=':',
                    label=f'Median Calendar: {median_calendar:.1f} days')
        ax5.axvline(max_calendar, color='green', linestyle='-',
                    label=f'Max Calendar: {max_calendar:.0f} days')

        # Customize drawdown duration distribution plot
        ax5.set_title('Drawdown Duration Distribution')
        ax5.set_xlabel('Duration (Days)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, linestyle='--', alpha=0.7)
    else:
        ax5.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown scatter (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])

    if durations and max_drawdowns:
        # Scatter plot with calendar days instead of trading days
        ax6.scatter(max_drawdowns, calendar_days, alpha=0.7, c='green', s=50, label='Calendar Days')
        ax6.scatter(max_drawdowns, durations, alpha=0.7, c='blue', s=30, label='Trading Days')

        # Add regression line for calendar days
        if len(calendar_days) > 1:
            z = np.polyfit(max_drawdowns, calendar_days, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(max_drawdowns), max(max_drawdowns), 100)
            ax6.plot(x_range, p(x_range), 'g--', alpha=0.7)

            # Calculate correlation
            corr = np.corrcoef(max_drawdowns, calendar_days)[0, 1]
            ax6.text(0.05, 0.95, f'Calendar Day Correlation: {corr:.2f}',
                     transform=ax6.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Customize scatter plot
        ax6.set_title('Drawdown Magnitude vs Duration')
        ax6.set_xlabel('Maximum Drawdown (%)')
        ax6.set_ylabel('Duration (Days)')
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Add overall title
    plt.suptitle(f'Drawdown Analysis ({test_start_date} to {test_end_date})', fontsize=16, y=0.99)

    # FIX: Replace tight_layout with more manual control of the layout
    # First adjust all subplots
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)

    # Then manually adjust for the suptitle
    fig.subplots_adjust(top=0.92)  # Reserve space for suptitle

    save_path = os.path.join(output_dir, f"{portfolio_name}_drawdown_analysis_{period_length}d.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return drawdown statistics with date information
    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown_depth,
        'total_drawdown_days': total_days_in_drawdown,
        'significant_drawdown_days': significant_drawdown_days,
        'avg_drawdown_length': avg_drawdown_length,
        'avg_calendar_days': avg_calendar_days,
        'drawdown_periods': len(drawdown_periods),
        'significant_periods': len(significant_periods),
        'max_drawdown_duration': max(period['duration'] for period in drawdown_periods) if drawdown_periods else 0,
        'max_calendar_duration': max(period['calendar_days'] for period in drawdown_periods) if drawdown_periods else 0,
        'drawdown_durations': [period['duration'] for period in drawdown_periods],
        'calendar_durations': [period['calendar_days'] for period in drawdown_periods],
        'drawdown_magnitudes': [period['max_drawdown'] for period in drawdown_periods],
        'top_significant_periods': significant_periods[:5]
    }


def plot_drawdown_distributions(simulation_results, actual_max_drawdown, actual_dd_duration, period_length, output_dir,
                                portfolio_name):
    """
    Plot side-by-side distributions of drawdown magnitude and duration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot drawdown magnitude distribution
    max_drawdowns = simulation_results['max_drawdowns']

    sns.histplot(max_drawdowns, kde=True, bins=30, ax=ax1, color='blue', alpha=0.6)
    ax1.axvline(x=actual_max_drawdown, color='r', linestyle='--',
                label=f'Actual: {actual_max_drawdown:.2f}%')

    # Add percentile information
    dd_percentile = stats.percentileofscore(max_drawdowns, actual_max_drawdown)
    ax1.text(0.05, 0.95, f'Actual Percentile: {dd_percentile:.1f}%',
             transform=ax1.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Calculate statistics
    dd_mean = np.mean(max_drawdowns)
    dd_median = np.median(max_drawdowns)
    dd_std = np.std(max_drawdowns)
    dd_5th = np.percentile(max_drawdowns, 5)
    dd_95th = np.percentile(max_drawdowns, 95)

    # Add statistics
    stats_text = (f'Mean: {dd_mean:.2f}%\n'
                  f'Median: {dd_median:.2f}%\n'
                  f'Std Dev: {dd_std:.2f}%\n'
                  f'5th %ile: {dd_5th:.2f}%\n'
                  f'95th %ile: {dd_95th:.2f}%')

    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Customize plot
    ax1.set_title(f'Maximum Drawdown Distribution - {period_length} Day Forward Test', fontsize=14)
    ax1.set_xlabel('Maximum Drawdown (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot drawdown duration distribution
    max_dd_durations = simulation_results['max_drawdown_durations']

    # For the simulated durations, estimate calendar days
    # Typically trading days are ~252/365 = 0.69 of calendar days, so we'll scale by 1.45
    estimated_calendar_durations = [d * 1.45 for d in max_dd_durations]

    sns.histplot(estimated_calendar_durations, kde=True, bins=30, ax=ax2, color='green', alpha=0.6,
                 label='Estimated Calendar Days')
    ax2.axvline(x=actual_dd_duration, color='r', linestyle='--',
                label=f'Actual: {actual_dd_duration} calendar days')

    # Add percentile information
    duration_percentile = stats.percentileofscore(estimated_calendar_durations, actual_dd_duration)
    ax2.text(0.05, 0.95, f'Actual Percentile: {duration_percentile:.1f}%',
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Calculate statistics
    dur_mean = np.mean(estimated_calendar_durations)
    dur_median = np.median(estimated_calendar_durations)
    dur_std = np.std(estimated_calendar_durations)
    dur_5th = np.percentile(estimated_calendar_durations, 5)
    dur_95th = np.percentile(estimated_calendar_durations, 95)

    # Add statistics
    stats_text = (f'Mean: {dur_mean:.1f} days\n'
                  f'Median: {dur_median:.1f} days\n'
                  f'Std Dev: {dur_std:.1f} days\n'
                  f'5th %ile: {dur_5th:.1f} days\n'
                  f'95th %ile: {dur_95th:.1f} days')

    ax2.text(0.95, 0.95, stats_text,
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Customize plot
    ax2.set_title(f'Maximum Drawdown Duration Distribution - {period_length} Day Forward Test', fontsize=14)
    ax2.set_xlabel('Maximum Drawdown Duration (Calendar Days)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{portfolio_name}_drawdown_distributions_{period_length}d.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return statistics
    return {
        'dd_mean': dd_mean,
        'dd_median': dd_median,
        'dd_std': dd_std,
        'dd_5th': dd_5th,
        'dd_95th': dd_95th,
        'dd_percentile': dd_percentile,
        'dur_mean': dur_mean,
        'dur_median': dur_median,
        'dur_std': dur_std,
        'dur_5th': dur_5th,
        'dur_95th': dur_95th,
        'dur_percentile': duration_percentile
    }


def run_walk_forward_test(dates, returns, test_period_length, output_dir, portfolio_name):
    """
    Run a walk-forward test where we use data up to a certain point to predict forward
    and compare with actual returns
    """
    if len(returns) <= test_period_length:
        print(f"Not enough data for walk-forward test of {test_period_length} days")
        return None

    # Split data into training and test sets
    train_returns = returns[:-test_period_length]
    test_returns = returns[-test_period_length:]
    test_dates = dates[-test_period_length:]

    # Only run simulation if we have enough training data
    if len(train_returns) < 30:  # Require at least 30 days for training
        print(f"Not enough training data for walk-forward test of {test_period_length} days")
        return None

    # Run simulation on training data
    print(f"\n--- Running Walk-Forward Test for {test_period_length} days ---")
    print(f"Training on {len(train_returns)} days, testing on {test_period_length} days")

    num_simulations = 10000
    simulation_results = run_monte_carlo_simulation(
        train_returns,
        num_simulations,
        test_period_length,
        annual_periods=252
    )

    # Calculate actual cumulative return path
    actual_returns = [0.0]  # Start with 0% return
    cumulative_return = 0.0

    for r in test_returns:
        # Convert daily percentage return to decimal
        r_decimal = r / 100.0

        # Calculate new cumulative return (compounded)
        cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
        actual_returns.append(cumulative_return)

    # Calculate actual metrics
    actual_final_return = actual_returns[-1]

    # Calculate actual drawdown statistics using our helper function
    test_start_date = dates[-test_period_length]
    test_end_date = dates[-1]

    drawdown_stats = analyze_drawdowns(
        actual_returns,
        output_dir,
        test_period_length,
        test_start_date,
        test_end_date,
        portfolio_name,
        dates=[test_dates[0]] + test_dates  # Add an initial date for the 0% return point
    )

    actual_max_drawdown = drawdown_stats['max_drawdown']
    actual_max_dd_duration = drawdown_stats['max_calendar_duration']

    # Plot drawdown distributions for simulations vs actual with portfolio name added
    if test_period_length >= 63:  # Only for periods of 3+ months
        dd_distribution_stats = plot_drawdown_distributions(
            simulation_results,
            actual_max_drawdown,
            actual_max_dd_duration,
            test_period_length,
            output_dir,
            portfolio_name
        )

    # Calculate annualized metrics if appropriate
    if test_period_length >= 20:  # Only calculate for meaningful periods
        actual_years = test_period_length / 252
        actual_annualized_return = ((1 + actual_final_return / 100) ** (1 / actual_years) - 1) * 100

        # Calculate actual Sharpe ratio
        actual_volatility = np.std(test_returns) * np.sqrt(252)
        actual_sharpe = (actual_annualized_return / actual_volatility) if actual_volatility != 0 else 0
    else:
        actual_annualized_return = actual_final_return  # For very short periods, use simple return
        actual_sharpe = 0

    # Get percentile rank of actual result within simulation
    final_returns = simulation_results['final_returns']
    actual_percentile = stats.percentileofscore(final_returns, actual_final_return)

    # Plot simulation with actual path overlaid
    plt.figure(figsize=(12, 8))

    # Plot percentile bands
    percentiles = simulation_results['percentiles']
    x = range(len(percentiles['50']))
    plt.fill_between(x, percentiles['5'], percentiles['95'], color='lightblue', alpha=0.3, label='5th-95th Percentile')
    plt.fill_between(x, percentiles['25'], percentiles['75'], color='blue', alpha=0.3, label='25th-75th Percentile')
    plt.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')

    # Find the best and worst paths based on final return
    all_paths = simulation_results['paths']
    best_path_idx = np.argmax(final_returns)
    worst_path_idx = np.argmin(final_returns)

    # Plot actual path
    plt.plot(x, actual_returns, 'orange', linewidth=3,
             label=f'Actual ({actual_final_return:.2f}%, {actual_percentile:.1f}%ile)')

    # Format plot
    if test_period_length <= 63:  # 3 months
        period_desc = f"{test_period_length} days (~3 months)"
    elif test_period_length <= 126:  # 6 months
        period_desc = f"{test_period_length} days (~6 months)"
    elif test_period_length <= 252:  # 1 year
        period_desc = f"{test_period_length} days (~1 year)"
    elif test_period_length <= 504:  # 2 years
        period_desc = f"{test_period_length} days (~2 years)"
    else:
        period_desc = f"{test_period_length} days"

    plt.title(f'Walk-Forward Test: {period_desc} ({test_start_date} to {test_end_date})', fontsize=14)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Save the figure
    save_path = os.path.join(output_dir, f"{portfolio_name}_walk_forward_{test_period_length}d.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # Generate CAGR distribution plot for longer periods
    if test_period_length >= 252:  # Only for 1-year or longer periods
        # Calculate CAGR for all simulations
        years = test_period_length / 252
        cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
        actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100

        plt.figure(figsize=(10, 6))
        sns.histplot(cagr_values, kde=True, bins=50)
        plt.axvline(x=actual_cagr, color='r', linestyle='--',
                    label=f'Actual CAGR: {actual_cagr:.2f}%')

        # Add percentile information
        cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
        plt.text(0.05, 0.95, f'Actual CAGR Percentile: {cagr_percentile:.1f}%',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f'CAGR Distribution - {period_desc} Forward Test', fontsize=14)
        plt.xlabel('CAGR (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the CAGR distribution plot
        cagr_plot_path = os.path.join(output_dir, f"{portfolio_name}_cagr_distribution_{test_period_length}d.png")
        plt.savefig(cagr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate statistics for CAGR distribution
        cagr_mean = np.mean(cagr_values)
        cagr_median = np.median(cagr_values)
        cagr_std = np.std(cagr_values)
        cagr_5th = np.percentile(cagr_values, 5)
        cagr_95th = np.percentile(cagr_values, 95)

        print(f"\nCAGR Distribution Statistics:")
        print(f"Mean CAGR: {cagr_mean:.2f}%")
        print(f"Median CAGR: {cagr_median:.2f}%")
        print(f"Standard Deviation: {cagr_std:.2f}%")
        print(f"5th Percentile: {cagr_5th:.2f}%")
        print(f"95th Percentile: {cagr_95th:.2f}%")
        print(f"Actual CAGR: {actual_cagr:.2f}% (Percentile: {cagr_percentile:.1f}%)")

    # Print summary statistics
    print(f"\nWalk-Forward Test Results for {period_desc}:")
    print(f"Test Period: {test_start_date} to {test_end_date}")
    print(f"Actual Cumulative Return: {actual_final_return:.2f}%")

    if test_period_length >= 20:
        print(f"Actual Annualized Return: {actual_annualized_return:.2f}%")
        print(f"Actual Sharpe Ratio: {actual_sharpe:.2f}")
        print(f"Actual Max Drawdown: {actual_max_drawdown:.2f}%")

    print(f"Percentile Rank: {actual_percentile:.1f}%")

    # Calculate forecast accuracy metrics
    median_return = percentiles['50'][-1]
    error = actual_final_return - median_return
    percent_error = (error / abs(median_return)) * 100 if abs(median_return) > 0.01 else 0

    print(f"Median Forecast: {median_return:.2f}%")
    print(f"Forecast Error: {error:.2f}% ({percent_error:.2f}%)")

    # Check if actual was within confidence intervals
    in_90_interval = percentiles['5'][-1] <= actual_final_return <= percentiles['95'][-1]
    in_50_interval = percentiles['25'][-1] <= actual_final_return <= percentiles['75'][-1]

    print(f"Actual within 90% Confidence Interval: {in_90_interval}")
    print(f"Actual within 50% Confidence Interval: {in_50_interval}")

    # Print drawdown statistics
    print(f"\nDrawdown Analysis:")
    print(f"Maximum Drawdown: {actual_max_drawdown:.2f}%")
    print(f"Average Drawdown: {drawdown_stats['avg_drawdown']:.2f}%")
    print(f"Total Days in Drawdown: {drawdown_stats['total_drawdown_days']} trading days")
    print(f"Number of Drawdown Periods: {drawdown_stats['drawdown_periods']}")
    print(f"Average Trading Day Length: {drawdown_stats['avg_drawdown_length']:.1f} days")
    print(f"Average Calendar Day Length: {drawdown_stats['avg_calendar_days']:.1f} days")
    print(
        f"Maximum Drawdown Duration: {drawdown_stats['max_drawdown_duration']} trading days, {drawdown_stats['max_calendar_duration']} calendar days")

    # Results dictionary with both trading and calendar days
    result = {
        'period_length': test_period_length,
        'test_start_date': test_start_date,
        'test_end_date': test_end_date,
        'actual_final_return': actual_final_return,
        'actual_annualized_return': actual_annualized_return if test_period_length >= 20 else None,
        'actual_sharpe': actual_sharpe if test_period_length >= 20 else None,
        'actual_max_drawdown': actual_max_drawdown,
        'actual_dd_duration_trading': drawdown_stats['max_drawdown_duration'],
        'actual_dd_duration_calendar': actual_max_dd_duration,
        'actual_percentile': actual_percentile,
        'median_forecast': median_return,
        'forecast_error': error,
        'percent_error': percent_error,
        'in_90_interval': in_90_interval,
        'in_50_interval': in_50_interval,
        # Add drawdown statistics
        'avg_drawdown': drawdown_stats['avg_drawdown'],
        'total_drawdown_days': drawdown_stats['total_drawdown_days'],
        'drawdown_periods': drawdown_stats['drawdown_periods'],
        'avg_drawdown_length_trading': drawdown_stats['avg_drawdown_length'],
        'avg_drawdown_length_calendar': drawdown_stats['avg_calendar_days']
    }

    # Add CAGR and drawdown distribution statistics for longer periods
    if test_period_length >= 252:
        result.update({
            'cagr_mean': cagr_mean,
            'cagr_median': cagr_median,
            'cagr_std': cagr_std,
            'cagr_5th': cagr_5th,
            'cagr_95th': cagr_95th,
            'actual_cagr': actual_cagr,
            'cagr_percentile': cagr_percentile
        })

    # Add drawdown distribution statistics for periods >= 3 months
    if test_period_length >= 63:
        result.update({
            'dd_mean': dd_distribution_stats['dd_mean'],
            'dd_median': dd_distribution_stats['dd_median'],
            'dd_std': dd_distribution_stats['dd_std'],
            'dd_5th': dd_distribution_stats['dd_5th'],
            'dd_95th': dd_distribution_stats['dd_95th'],
            'dd_percentile': dd_distribution_stats['dd_percentile'],
            'dur_mean': dd_distribution_stats['dur_mean'],
            'dur_median': dd_distribution_stats['dur_median'],
            'dur_std': dd_distribution_stats['dur_std'],
            'dur_5th': dd_distribution_stats['dur_5th'],
            'dur_95th': dd_distribution_stats['dur_95th'],
            'dur_percentile': dd_distribution_stats['dur_percentile']
        })

    return result


def run_expanding_window_test(dates, returns, initial_train_period=252, test_period_length=252,
                              expansion_size=252, output_dir=None, portfolio_name=None):
    """
    Run walk-forward tests using an expanding window approach.

    Parameters:
    -----------
    dates : list
        List of date strings for the full dataset
    returns : list
        List of daily returns for the full dataset
    initial_train_period : int, optional
        Length of the initial training period in trading days (default: 252, ~1 year)
    test_period_length : int, optional
        Length of each test period in trading days (default: 252, ~1 year)
    expansion_size : int, optional
        Number of days to expand the training window by in each iteration (default: 252, ~1 year)
    output_dir : str
        Directory to save output files
    portfolio_name : str
        Name of the portfolio for file naming
    """
    # Check if we have enough data for at least one iteration
    if len(returns) < (initial_train_period + test_period_length):
        print(
            f"Not enough data for expanding window test. Need at least {initial_train_period + test_period_length} days.")
        return None

    # Create a specific directory for expanding window results
    expanding_dir = os.path.join(output_dir, f"{portfolio_name}_expanding_window")
    os.makedirs(expanding_dir, exist_ok=True)

    # Determine how many iterations we can run
    available_days = len(returns) - initial_train_period
    num_iterations = available_days // test_period_length

    if num_iterations == 0:
        print(f"Not enough data for any complete test periods after initial training window.")
        return None

    print(f"\n--- Running Expanding Window Test ---")
    print(f"Initial training period: {initial_train_period} days")
    print(f"Test period: {test_period_length} days")
    print(f"Training window expansion: {expansion_size} days per iteration")
    print(f"Number of iterations: {num_iterations}")

    # Lists to store results
    period_labels = []
    train_sizes = []
    actual_returns = []
    forecast_returns = []
    actual_cagrs = []
    forecast_cagrs = []
    max_drawdowns = []
    actual_percentiles = []

    # Run iterations
    for i in range(num_iterations):
        # Calculate indices for this iteration
        train_start_idx = 0  # Always start from the beginning of the data

        # Calculate the size of the training window for this iteration
        # Initial window size + expansion for each completed iteration
        train_size = initial_train_period + (i * expansion_size)
        train_end_idx = train_size

        # Test window starts right after training window
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_period_length, len(returns))

        # Check if we have enough data for this test period
        if test_end_idx - test_start_idx < 5:  # Require at least 5 days of test data
            print(f"Skipping iteration {i + 1} - not enough data for meaningful test period")
            continue

        # Extract data for this iteration
        train_data = returns[train_start_idx:train_end_idx]
        test_data = returns[test_start_idx:test_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_start_date = dates[train_start_idx]
        train_end_date = dates[train_end_idx - 1]
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]

        period_label = f"{test_start_date} to {test_end_date}"
        period_labels.append(period_label)
        train_sizes.append(train_size)

        print(f"\nIteration {i + 1} of {num_iterations}:")
        print(f"Training period: {train_start_date} to {train_end_date} ({len(train_data)} days)")
        print(f"Test period: {test_start_date} to {test_end_date} ({len(test_data)} days)")

        # Run Monte Carlo simulation on the training data
        num_simulations = 10000
        simulation_results = run_monte_carlo_simulation(
            train_data,
            num_simulations,
            len(test_data),
            annual_periods=252
        )

        # Calculate actual cumulative return path
        actual_return_path = [0.0]  # Start with 0% return
        cumulative_return = 0.0

        for r in test_data:
            # Convert daily percentage return to decimal
            r_decimal = r / 100.0

            # Calculate new cumulative return (compounded)
            cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
            actual_return_path.append(cumulative_return)

        actual_final_return = actual_return_path[-1]
        actual_returns.append(actual_final_return)

        # Get forecast (median) from simulation
        median_return = simulation_results['percentiles']['50'][-1]
        forecast_returns.append(median_return)

        # Calculate actual vs forecasted CAGR for meaningful periods
        if len(test_data) >= 20:
            actual_years = len(test_data) / 252
            actual_annualized_return = ((1 + actual_final_return / 100) ** (1 / actual_years) - 1) * 100
            forecast_annualized_return = ((1 + median_return / 100) ** (1 / actual_years) - 1) * 100

            actual_cagrs.append(actual_annualized_return)
            forecast_cagrs.append(forecast_annualized_return)

        # Calculate actual drawdown statistics
        drawdown_stats = analyze_drawdowns(
            actual_return_path,
            expanding_dir,
            len(test_data),
            test_start_date,
            test_end_date,
            f"{portfolio_name}_expanding_iter{i + 1}",
            dates=[test_dates[0]] + test_dates  # Add an initial date for the 0% return point
        )

        max_drawdowns.append(drawdown_stats['max_drawdown'])

        # Get percentile rank of actual result within simulation
        final_returns = simulation_results['final_returns']
        percentile = stats.percentileofscore(final_returns, actual_final_return)
        actual_percentiles.append(percentile)

        # Save the Monte Carlo simulation plot with actual path
        plt.figure(figsize=(12, 8))

        # Plot percentile bands
        percentiles = simulation_results['percentiles']
        x = range(len(percentiles['50']))
        plt.fill_between(x, percentiles['5'], percentiles['95'], color='lightblue', alpha=0.3,
                         label='5th-95th Percentile')
        plt.fill_between(x, percentiles['25'], percentiles['75'], color='blue', alpha=0.3, label='25th-75th Percentile')
        plt.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')

        # Plot actual path
        plt.plot(x, actual_return_path, 'orange', linewidth=3,
                 label=f'Actual ({actual_final_return:.2f}%, {percentile:.1f}%ile)')

        # Format plot
        plt.title(f'Expanding Window Test: Iteration {i + 1} ({test_start_date} to {test_end_date})', fontsize=14)
        plt.xlabel('Trading Days', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        # Add training size information
        plt.text(0.02, 0.02, f'Training Size: {train_size} days ({train_start_date} to {train_end_date})',
                 transform=plt.gca().transAxes, fontsize=10,
                 horizontalalignment='left', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save the figure
        save_path = os.path.join(expanding_dir, f"{portfolio_name}_expanding_iter{i + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.close()

        print(f"Actual Return: {actual_final_return:.2f}%, Forecast: {median_return:.2f}%")
        print(f"Percentile: {percentile:.1f}%, Max Drawdown: {drawdown_stats['max_drawdown']:.2f}%")

        if len(actual_cagrs) > 0 and i < len(actual_cagrs):
            print(f"Actual CAGR: {actual_cagrs[-1]:.2f}%, Forecast CAGR: {forecast_cagrs[-1]:.2f}%")

    # Create summary visualizations
    if len(period_labels) > 0:
        # Create date labels for x-axis (convert to month/year format)
        date_labels = []
        for period in period_labels:
            start_date = period.split(" to ")[0]  # Extract start date from "YYYY-MM-DD to YYYY-MM-DD"
            try:
                dt = datetime.strptime(start_date, '%Y-%m-%d')
                date_labels.append(dt.strftime('%b %Y'))  # Format as "Jan 2023"
            except:
                # Fallback to iteration number if date conversion fails
                date_labels.append(f"Period {len(date_labels) + 1}")

        # Create a plot showing how forecast accuracy changes with training size
        plt.figure(figsize=(14, 7))

        # Calculate forecast errors
        forecast_errors = [a - f for a, f in zip(actual_returns, forecast_returns)]
        abs_errors = [abs(err) for err in forecast_errors]

        # Create scatter plot with training size on x-axis and absolute error on y-axis
        plt.scatter(train_sizes, abs_errors, s=80, alpha=0.7, c=train_sizes, cmap='viridis')

        # Fit a trend line
        if len(train_sizes) > 1:
            z = np.polyfit(train_sizes, abs_errors, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(train_sizes), max(train_sizes), 100)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2)

            # Add correlation text
            corr = np.corrcoef(train_sizes, abs_errors)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.2f}',
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Customize plot
        plt.title('Forecast Error vs Training Window Size', fontsize=14)
        plt.xlabel('Training Window Size (Trading Days)', fontsize=12)
        plt.ylabel('Absolute Forecast Error (%)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add test period labels to points
        for i, (x, y, lbl) in enumerate(zip(train_sizes, abs_errors, date_labels)):
            plt.annotate(lbl, (x, y), xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(os.path.join(expanding_dir, f"{portfolio_name}_expanding_error_vs_size.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Create a plot comparing actual vs forecast returns
        plt.figure(figsize=(14, 7))

        # Set up index for bars
        indices = np.arange(len(period_labels))
        width = 0.35

        # Create bar chart of actual vs forecasted returns
        plt.bar(indices - width / 2, actual_returns, width, label='Actual Return', color='green', alpha=0.7)
        plt.bar(indices + width / 2, forecast_returns, width, label='Forecast Return', color='blue', alpha=0.7)

        # Add value labels on bars
        for i, v in enumerate(actual_returns):
            plt.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

        for i, v in enumerate(forecast_returns):
            plt.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

        # Customize plot
        plt.xlabel('Test Period Start')
        plt.ylabel('Cumulative Return (%)')
        plt.title(f'Expanding Window Test: Actual vs Forecast Returns - {portfolio_name}')
        plt.xticks(indices, date_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()

        # Add percentile labels
        for i, pct in enumerate(actual_percentiles):
            plt.text(i, max(actual_returns[i], forecast_returns[i]) + 5, f"{pct:.0f}%ile",
                     ha='center', fontsize=9, color='red')

        # Add training window size labels
        for i, size in enumerate(train_sizes):
            plt.text(i, min(actual_returns[i], forecast_returns[i]) - 5, f"{size} days",
                     ha='center', fontsize=8, color='blue')

        plt.tight_layout()
        plt.savefig(os.path.join(expanding_dir, f"{portfolio_name}_expanding_returns_comparison.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Create CAGR comparison chart if we have CAGR data
        if len(actual_cagrs) > 0:
            plt.figure(figsize=(14, 7))

            # Create bar chart of actual vs forecasted CAGR
            plt.bar(indices - width / 2, actual_cagrs, width, label='Actual CAGR', color='green', alpha=0.7)
            plt.bar(indices + width / 2, forecast_cagrs, width, label='Forecast CAGR', color='blue', alpha=0.7)

            # Add value labels on bars
            for i, v in enumerate(actual_cagrs):
                plt.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

            for i, v in enumerate(forecast_cagrs):
                plt.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

            # Customize plot
            plt.xlabel('Test Period Start')
            plt.ylabel('Annualized Return (%)')
            plt.title(f'Expanding Window Test: Actual vs Forecast CAGR - {portfolio_name}')
            plt.xticks(indices, date_labels, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(expanding_dir, f"{portfolio_name}_expanding_cagr_comparison.png"), dpi=300,
                        bbox_inches='tight')
            plt.close()

            # Create a plot of max drawdowns
            plt.figure(figsize=(14, 7))
            plt.bar(indices, max_drawdowns, color='red', alpha=0.7)

            # Add value labels on bars
            for i, v in enumerate(max_drawdowns):
                plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

            # Customize plot
            plt.xlabel('Test Period Start')
            plt.ylabel('Maximum Drawdown (%)')
            plt.title(f'Expanding Window Test: Maximum Drawdowns - {portfolio_name}')
            plt.xticks(indices, date_labels, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(os.path.join(expanding_dir, f"{portfolio_name}_expanding_drawdowns.png"), dpi=300,
                        bbox_inches='tight')
            plt.close()

            # Create a DataFrame with all results
            results_data = {
                'Iteration': list(range(1, len(period_labels) + 1)),
                'Period': period_labels,
                'Training_Size': train_sizes,
                'Actual_Return': actual_returns,
                'Forecast_Return': forecast_returns,
                'Error': [a - f for a, f in zip(actual_returns, forecast_returns)],
                'Abs_Error': [abs(a - f) for a, f in zip(actual_returns, forecast_returns)],
                'Percentile': actual_percentiles,
                'Max_Drawdown': max_drawdowns
            }

            if len(actual_cagrs) > 0:
                results_data['Actual_CAGR'] = actual_cagrs
                results_data['Forecast_CAGR'] = forecast_cagrs
                results_data['CAGR_Error'] = [a - f for a, f in zip(actual_cagrs, forecast_cagrs)]

            results_df = pd.DataFrame(results_data)

            # Save results to CSV
            csv_path = os.path.join(expanding_dir, f"{portfolio_name}_expanding_results.csv")
            results_df.to_csv(csv_path, index=False)

            # Print summary statistics
            print("\nExpanding Window Test Summary:")
            print(f"Average Actual Return: {np.mean(actual_returns):.2f}%")
            print(f"Average Forecast Return: {np.mean(forecast_returns):.2f}%")
            print(f"Average Error: {np.mean([a - f for a, f in zip(actual_returns, forecast_returns)]):.2f}%")
            print(
                f"Average Absolute Error: {np.mean([abs(a - f) for a, f in zip(actual_returns, forecast_returns)]):.2f}%")
            print(f"Average Percentile: {np.mean(actual_percentiles):.1f}%")
            print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2f}%")

            if len(actual_cagrs) > 0:
                print(f"Average Actual CAGR: {np.mean(actual_cagrs):.2f}%")
                print(f"Average Forecast CAGR: {np.mean(forecast_cagrs):.2f}%")

            # Check if accuracy improves with training size
            if len(train_sizes) > 1:
                corr = np.corrcoef(train_sizes, abs_errors)[0, 1]
                trend = "improves" if corr < 0 else "worsens" if corr > 0 else "doesn't change"
                print(f"\nForecast accuracy {trend} with increasing training window size (correlation: {corr:.2f})")

            print(f"\nExpanding window test results saved to: {expanding_dir}/")

            return results_df

        return None

def run_rolling_walk_forward_test(dates, returns, train_period_length=None, test_period_length=None,
                                  output_dir=None, portfolio_name=None, step_size=None,
                                  fixed_train_start_date=None, fixed_train_end_date=None,
                                  extend_last_to_current=True, allow_overlap=False):
    """
    Run multiple walk-forward tests by rolling through the historical data

    Parameters:
    -----------
    dates : list
        List of date strings for the full dataset
    returns : list
        List of daily returns for the full dataset
    train_period_length : int, optional
        Length of the training period in trading days (used if fixed_train_start_date is None)
    test_period_length : int
        Length of the test period in trading days
    output_dir : str
        Directory to save output files
    portfolio_name : str
        Name of the portfolio for file naming
    step_size : int, optional
        Number of days to step forward for each iteration (defaults to test_period_length)
    fixed_train_start_date : str, optional
        Start date for fixed training window (format: 'YYYY-MM-DD')
    fixed_train_end_date : str, optional
        End date for fixed training window (format: 'YYYY-MM-DD')
    extend_last_to_current : bool, optional
        Whether to extend the last iteration to include the most current data
    allow_overlap : bool, optional
        Whether to allow test windows to overlap with the training window
        When True and using fixed training window, test windows start from the beginning of the data
    """
    # Convert dates to datetime objects for easier comparison
    date_objects = [pd.to_datetime(d) if isinstance(d, str) else d for d in dates]

    # Determine if we're using a fixed training window or a sliding window
    using_fixed_window = (fixed_train_start_date is not None and fixed_train_end_date is not None)

    if using_fixed_window:
        # Convert fixed training dates to datetime
        fixed_start = pd.to_datetime(fixed_train_start_date)
        fixed_end = pd.to_datetime(fixed_train_end_date)

        # Find indices for fixed training window
        train_start_idx = None
        train_end_idx = None

        for i, date in enumerate(date_objects):
            if train_start_idx is None and date >= fixed_start:
                train_start_idx = i
            if train_end_idx is None and date > fixed_end:  # Use > to get the day after end date
                train_end_idx = i
                break

        # Handle edge cases
        if train_start_idx is None:
            print(f"Fixed training start date {fixed_train_start_date} is after all available data")
            return None
        if train_end_idx is None:
            train_end_idx = len(date_objects)  # Use all data up to end

        # If allow_overlap is True, set the test start index to the beginning of the data
        if allow_overlap:
            test_start_idx = 0
            print(f"Using fixed training window from {dates[train_start_idx]} to {dates[train_end_idx - 1]}")
            print(f"Training data size: {train_end_idx - train_start_idx} days")
            print(f"Test windows will start from the beginning of the historical record ({dates[0]})")
        else:
            # Traditional approach - test windows start after training window
            test_start_idx = train_end_idx

            # Check if we have enough data for at least one test period
            if len(date_objects) - test_start_idx < test_period_length:
                print(f"Not enough data after fixed training window for test period of {test_period_length} days")
                return None

            print(f"Using fixed training window from {dates[train_start_idx]} to {dates[train_end_idx - 1]}")
            print(f"Training data size: {train_end_idx - train_start_idx} days")
            print(f"Test windows will start after training window ({dates[test_start_idx]})")
    else:
        # Using sliding window approach
        if len(returns) < (train_period_length + test_period_length) and not allow_overlap:
            print(
                f"Not enough data for non-overlapping rolling walk-forward test. Need at least {train_period_length + test_period_length} days.")
            return None

    # If step_size is not specified, default to test_period_length (non-overlapping periods)
    if step_size is None:
        step_size = test_period_length

    # Create a specific directory for rolling walk results
    rolling_dir = os.path.join(output_dir, f"{portfolio_name}_rolling_walk")
    os.makedirs(rolling_dir, exist_ok=True)

    # Determine how many iterations we can run and where to start tests
    if using_fixed_window:
        if allow_overlap:
            # When allowing overlap with fixed window, we start from the beginning
            # and run until the end of the data
            available_test_days = len(returns)
            test_start_position = 0
        else:
            # Traditional approach - test after training
            available_test_days = len(returns) - train_end_idx
            test_start_position = train_end_idx
    else:
        if allow_overlap:
            # If allowing overlap with sliding window, we can also start from beginning
            available_test_days = len(returns)
            test_start_position = 0
        else:
            # Traditional approach - test after training
            available_test_days = len(returns) - train_period_length
            test_start_position = train_period_length

    num_iterations = max(1, (available_test_days - test_period_length) // step_size + 1)

    print(f"\n--- Running Rolling Walk-Forward Test ---")
    if using_fixed_window:
        print(f"Fixed training window: {fixed_train_start_date} to {fixed_train_end_date}")
    else:
        print(f"Sliding training period: {train_period_length} days")
    print(f"Test period: {test_period_length} days")
    print(f"Step size: {step_size} days")
    print(f"Number of iterations: {num_iterations}")
    if extend_last_to_current:
        print(f"Last iteration will be extended to include most recent data")
    if allow_overlap:
        print(f"Warning: Test windows allowed to overlap with training window")

    # Lists to store results
    period_labels = []
    actual_returns = []
    forecast_returns = []
    actual_cagrs = []
    forecast_cagrs = []
    max_drawdowns = []
    actual_percentiles = []

    # Run iterations
    for i in range(num_iterations):
        # Determine indices for this iteration
        if using_fixed_window:
            # Fixed training window stays the same for all iterations
            start_idx = train_start_idx
            train_end_idx_i = train_end_idx

            # Test window depends on overlap setting
            if allow_overlap:
                test_start_idx = test_start_position + (i * step_size)
            else:
                test_start_idx = train_end_idx + (i * step_size)

            # For the last iteration, extend to the end of available data if requested
            if i == num_iterations - 1 and extend_last_to_current:
                test_end_idx = len(returns)
            else:
                test_end_idx = min(test_start_idx + test_period_length, len(returns))
        else:
            # For sliding window approach
            if allow_overlap:
                # Test windows can start from the beginning
                test_start_idx = test_start_position + (i * step_size)
                start_idx = max(0, test_start_idx - train_period_length)  # Training window ends right before test
            else:
                # Standard sliding window approach
                start_idx = i * step_size
                train_end_idx_i = start_idx + train_period_length
                test_start_idx = train_end_idx_i

            # Training window ends right before test window starts, unless we're allowing overlap
            train_end_idx_i = test_start_idx if not allow_overlap else (start_idx + train_period_length)

            # For the last iteration, extend to the end of available data if requested
            if i == num_iterations - 1 and extend_last_to_current:
                test_end_idx = len(returns)
            else:
                test_end_idx = min(test_start_idx + test_period_length, len(returns))

        # Check if we have enough data for this test period
        if test_end_idx - test_start_idx < 5:  # Require at least 5 days of test data
            print(f"Skipping iteration {i + 1} - not enough data for meaningful test period")
            continue

        # Extract data for this iteration
        train_data = returns[start_idx:train_end_idx_i]
        test_data = returns[test_start_idx:test_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_start_date = dates[start_idx]
        train_end_date = dates[train_end_idx_i - 1]
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]

        period_label = f"{test_start_date} to {test_end_date}"
        period_labels.append(period_label)

        # Check for overlap between training and test data
        has_overlap = (test_start_idx < train_end_idx_i)
        overlap_warning = " (OVERLAPS TRAINING)" if has_overlap else ""

        print(f"\nIteration {i + 1} of {num_iterations}:")
        print(f"Training period: {train_start_date} to {train_end_date} ({len(train_data)} days)")
        print(f"Test period: {test_start_date} to {test_end_date} ({len(test_data)} days){overlap_warning}")

        # Special message for extended last iteration
        if i == num_iterations - 1 and extend_last_to_current and len(test_data) > test_period_length:
            print(f"Note: Last iteration extended to include all current data ({len(test_data)} days)")

        # Run Monte Carlo simulation on the training data
        num_simulations = 10000
        simulation_results = run_monte_carlo_simulation(
            train_data,
            num_simulations,
            len(test_data),
            annual_periods=252
        )

        # Calculate actual cumulative return path
        actual_return_path = [0.0]  # Start with 0% return
        cumulative_return = 0.0

        for r in test_data:
            # Convert daily percentage return to decimal
            r_decimal = r / 100.0

            # Calculate new cumulative return (compounded)
            cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
            actual_return_path.append(cumulative_return)

        actual_final_return = actual_return_path[-1]
        actual_returns.append(actual_final_return)

        # Get forecast (median) from simulation
        median_return = simulation_results['percentiles']['50'][-1]
        forecast_returns.append(median_return)

        # Calculate actual vs forecasted CAGR for meaningful periods
        if len(test_data) >= 20:
            actual_years = len(test_data) / 252
            actual_annualized_return = ((1 + actual_final_return / 100) ** (1 / actual_years) - 1) * 100
            forecast_annualized_return = ((1 + median_return / 100) ** (1 / actual_years) - 1) * 100

            actual_cagrs.append(actual_annualized_return)
            forecast_cagrs.append(forecast_annualized_return)

        # Calculate actual drawdown statistics using our helper function
        drawdown_stats = analyze_drawdowns(
            actual_return_path,
            rolling_dir,
            len(test_data),
            test_start_date,
            test_end_date,
            f"{portfolio_name}_iter{i + 1}",
            dates=[test_dates[0]] + test_dates  # Add an initial date for the 0% return point
        )

        max_drawdowns.append(drawdown_stats['max_drawdown'])

        # Get percentile rank of actual result within simulation
        final_returns = simulation_results['final_returns']
        percentile = stats.percentileofscore(final_returns, actual_final_return)
        actual_percentiles.append(percentile)

        # Save the Monte Carlo simulation plot with actual path
        plt.figure(figsize=(12, 8))

        # Plot percentile bands
        percentiles = simulation_results['percentiles']
        x = range(len(percentiles['50']))
        plt.fill_between(x, percentiles['5'], percentiles['95'], color='lightblue', alpha=0.3,
                         label='5th-95th Percentile')
        plt.fill_between(x, percentiles['25'], percentiles['75'], color='blue', alpha=0.3, label='25th-75th Percentile')
        plt.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')

        # Plot actual path
        plt.plot(x, actual_return_path, 'orange', linewidth=3,
                 label=f'Actual ({actual_final_return:.2f}%, {percentile:.1f}%ile)')

        # Format plot
        plot_title = f'Rolling Walk-Forward Test: Iteration {i + 1} ({test_start_date} to {test_end_date})'
        if has_overlap:
            plot_title += f" - Overlaps Training"
        if i == num_iterations - 1 and extend_last_to_current and len(test_data) > test_period_length:
            plot_title += f" - Extended to Current Date"

        plt.title(plot_title, fontsize=14)
        plt.xlabel('Trading Days', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        # Save the figure
        save_path = os.path.join(rolling_dir, f"{portfolio_name}_rolling_iter{i + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.close()

        print(f"Actual Return: {actual_final_return:.2f}%, Forecast: {median_return:.2f}%")
        print(f"Percentile: {percentile:.1f}%, Max Drawdown: {drawdown_stats['max_drawdown']:.2f}%")

        if len(actual_cagrs) > 0 and i < len(actual_cagrs):
            print(f"Actual CAGR: {actual_cagrs[-1]:.2f}%, Forecast CAGR: {forecast_cagrs[-1]:.2f}%")

    # Create summary visualizations
    if len(period_labels) > 0:
        # Create date labels for x-axis (convert to month/year format)
        date_labels = []
        for period in period_labels:
            start_date = period.split(" to ")[0]  # Extract start date from "YYYY-MM-DD to YYYY-MM-DD"
            try:
                dt = datetime.strptime(start_date, '%Y-%m-%d')
                date_labels.append(dt.strftime('%b %Y'))  # Format as "Jan 2023"
            except:
                # Fallback to iteration number if date conversion fails
                date_labels.append(f"Period {len(date_labels) + 1}")

        # Create a plot comparing actual vs forecast returns
        plt.figure(figsize=(14, 7))

        # Set up index for bars
        indices = np.arange(len(period_labels))
        width = 0.35

        # Create bar chart of actual vs forecasted returns
        plt.bar(indices - width / 2, actual_returns, width, label='Actual Return', color='green', alpha=0.7)
        plt.bar(indices + width / 2, forecast_returns, width, label='Forecast Return', color='blue', alpha=0.7)

        # Add value labels on bars
        for i, v in enumerate(actual_returns):
            plt.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

        for i, v in enumerate(forecast_returns):
            plt.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

        # Customize plot
        plt.xlabel('Test Period Start')
        plt.ylabel('Cumulative Return (%)')
        title_prefix = "Fixed Window" if using_fixed_window else "Rolling Window"
        plt.title(f'{title_prefix} Walk-Forward Test: Actual vs Forecast Returns - {portfolio_name}')
        plt.xticks(indices, date_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()

        # Add percentile labels
        for i, pct in enumerate(actual_percentiles):
            plt.text(i, max(actual_returns[i], forecast_returns[i]) + 5, f"{pct:.0f}%ile",
                     ha='center', fontsize=9, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(rolling_dir, f"{portfolio_name}_rolling_returns_comparison.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Create CAGR comparison chart if we have CAGR data
        if len(actual_cagrs) > 0:
            plt.figure(figsize=(14, 7))

            # Create bar chart of actual vs forecasted CAGR
            plt.bar(indices - width / 2, actual_cagrs, width, label='Actual CAGR', color='green', alpha=0.7)
            plt.bar(indices + width / 2, forecast_cagrs, width, label='Forecast CAGR', color='blue', alpha=0.7)

            # Add value labels on bars
            for i, v in enumerate(actual_cagrs):
                plt.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

            for i, v in enumerate(forecast_cagrs):
                plt.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, rotation=90 if abs(v) > 20 else 0)

            # Customize plot
            plt.xlabel('Test Period Start')
            plt.ylabel('Annualized Return (%)')
            plt.title(f'{title_prefix} Walk-Forward Test: Actual vs Forecast CAGR - {portfolio_name}')
            plt.xticks(indices, date_labels, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(rolling_dir, f"{portfolio_name}_rolling_cagr_comparison.png"), dpi=300,
                        bbox_inches='tight')
            plt.close()

        # Create a plot of max drawdowns
        plt.figure(figsize=(14, 7))
        plt.bar(indices, max_drawdowns, color='red', alpha=0.7)

        # Add value labels on bars
        for i, v in enumerate(max_drawdowns):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

        # Customize plot
        plt.xlabel('Test Period Start')
        plt.ylabel('Maximum Drawdown (%)')
        plt.title(f'{title_prefix} Walk-Forward Test: Maximum Drawdowns - {portfolio_name}')
        plt.xticks(indices, date_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(rolling_dir, f"{portfolio_name}_rolling_drawdowns.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create a DataFrame with all results
        results_data = {
            'Iteration': list(range(1, len(period_labels) + 1)),
            'Period': period_labels,
            'Actual_Return': actual_returns,
            'Forecast_Return': forecast_returns,
            'Error': [a - f for a, f in zip(actual_returns, forecast_returns)],
            'Percentile': actual_percentiles,
            'Max_Drawdown': max_drawdowns
        }

        if len(actual_cagrs) > 0:
            results_data['Actual_CAGR'] = actual_cagrs
            results_data['Forecast_CAGR'] = forecast_cagrs
            results_data['CAGR_Error'] = [a - f for a, f in zip(actual_cagrs, forecast_cagrs)]

        results_df = pd.DataFrame(results_data)

        # Save results to CSV
        csv_path = os.path.join(rolling_dir, f"{portfolio_name}_rolling_results.csv")
        results_df.to_csv(csv_path, index=False)

        # Print summary statistics
        print("\nRolling Walk-Forward Test Summary:")
        print(f"Average Actual Return: {np.mean(actual_returns):.2f}%")
        print(f"Average Forecast Return: {np.mean(forecast_returns):.2f}%")
        print(f"Average Error: {np.mean([a - f for a, f in zip(actual_returns, forecast_returns)]):.2f}%")
        print(f"Average Percentile: {np.mean(actual_percentiles):.1f}%")
        print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2f}%")

        if len(actual_cagrs) > 0:
            print(f"Average Actual CAGR: {np.mean(actual_cagrs):.2f}%")
            print(f"Average Forecast CAGR: {np.mean(forecast_cagrs):.2f}%")

        print(f"\nRolling walk-forward test results saved to: {rolling_dir}/")

        return results_df

    return None


def add_forward_forecast_mode(dates, returns, output_dir, portfolio_name):
    """
    Generate a forward-looking forecast starting from the last day in the dataset
    with user-selectable training range and display random sample paths.

    Parameters:
    -----------
    dates : list
        List of date strings for the full dataset
    returns : list
        List of daily returns for the full dataset
    output_dir : str
        Directory to save output files
    portfolio_name : str
        Name of the portfolio for file naming
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    import os
    import random

    # Create a specific directory for forward forecast results
    forecast_dir = os.path.join(output_dir, f"{portfolio_name}_forward_forecast")
    os.makedirs(forecast_dir, exist_ok=True)

    print("\n------------------------------------------")
    print("Forward Forecast Mode")
    print("------------------------------------------")

    # Show available date range
    print(f"\nAvailable date range: {dates[0]} to {dates[-1]}")

    # Ask for training period start date
    default_start = dates[0]
    train_start_date = input(f"Enter training start date (default: {default_start}): ")
    train_start_date = train_start_date if train_start_date else default_start

    # Convert training start date to datetime for comparison
    train_start_dt = pd.to_datetime(train_start_date)

    # Ask for training period length as an alternative
    use_lookback = input("Would you like to specify a lookback period instead? (y/n, default: n): ").lower() == 'y'

    if use_lookback:
        default_lookback = 252  # Default to 1 year
        lookback_days = input(f"Enter lookback period in trading days (default: {default_lookback}): ")
        lookback_days = int(lookback_days) if lookback_days else default_lookback

        # Calculate training start date based on lookback
        all_dates = pd.to_datetime(dates)
        last_date = all_dates[-1]

        # Find index that's approximately lookback_days from the end
        if lookback_days >= len(all_dates):
            train_start_dt = all_dates[0]
            print(f"Lookback period exceeds available data. Using all available data.")
        else:
            train_start_dt = all_dates[-lookback_days]

        # Update the displayed training start date
        train_start_date = train_start_dt.strftime('%Y-%m-%d')
        print(f"Training will start from: {train_start_date}")

    # Ask for forecast period length
    default_forecast = 126  # Default to ~6 months
    forecast_period = input(f"Enter forecast period length in trading days (default: {default_forecast}): ")
    forecast_period = int(forecast_period) if forecast_period else default_forecast

    # Number of sample paths to show
    default_samples = 200
    num_samples = input(f"Enter number of sample paths to display (default: {default_samples}): ")
    num_samples = int(num_samples) if num_samples else default_samples

    # Filter training data based on start date
    date_objects = [pd.to_datetime(d) for d in dates]
    train_data = []
    train_dates = []

    for i, date_obj in enumerate(date_objects):
        if date_obj >= train_start_dt:
            train_data.append(returns[i])
            train_dates.append(dates[i])

    if len(train_data) < 30:  # Require at least 30 days for training
        print(f"Error: Not enough training data. Found only {len(train_data)} days after {train_start_date}.")
        print("Please select an earlier start date or use a shorter lookback period.")
        return

    print(f"\nTraining on {len(train_data)} days from {train_dates[0]} to {train_dates[-1]}")
    print(f"Generating forecast for {forecast_period} trading days forward")

    # Calculate approximate end date for forecast
    # Assuming ~252 trading days per year (~21 trading days per month)
    last_date_dt = pd.to_datetime(dates[-1])
    forecast_months = forecast_period / 21
    # Rough approximation: add calendar days assuming weekends and some holidays
    forecast_calendar_days = int(forecast_period * 1.4)  # ~1.4 calendar days per trading day
    forecast_end_date = (last_date_dt + timedelta(days=forecast_calendar_days)).strftime('%Y-%m-%d')

    print(f"Approximate forecast end date: {forecast_end_date}")

    # Run Monte Carlo simulation for the forecast period
    num_simulations = 10000
    annual_periods = 252

    print("\nRunning Monte Carlo simulation...")
    simulation_results = run_monte_carlo_simulation(
        train_data,
        num_simulations,
        forecast_period,
        annual_periods=annual_periods
    )

    # Extract forecast percentiles and statistics
    percentiles = simulation_results['percentiles']
    final_returns = simulation_results['final_returns']
    max_drawdowns = simulation_results['max_drawdowns']
    all_paths = simulation_results['paths']

    # Calculate summary statistics
    final_50th = percentiles['50'][-1]
    final_5th = percentiles['5'][-1]
    final_25th = percentiles['25'][-1]
    final_75th = percentiles['75'][-1]
    final_95th = percentiles['95'][-1]

    # Calculate CAGR statistics if forecast period is meaningful
    cagr_values = None
    if forecast_period >= 60:  # Only calculate for periods >= ~3 months
        years = forecast_period / annual_periods
        cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
        cagr_median = np.median(cagr_values)
        cagr_mean = np.mean(cagr_values)
        cagr_5th = np.percentile(cagr_values, 5)
        cagr_95th = np.percentile(cagr_values, 95)

    # Create the forecast path plot with sample paths
    plt.figure(figsize=(12, 8))
    x = range(len(percentiles['50']))

    # Select random sample paths
    num_paths = all_paths.shape[0]
    sample_indices = random.sample(range(num_paths), min(num_samples, num_paths))

    # Plot random sample paths with very light opacity
    for idx in sample_indices:
        path = all_paths[idx, :]
        plt.plot(x, path, color='gray', alpha=0.3, linewidth=0.5)

    # Add a label for the sample paths
    # Create a custom line for the legend (since individual paths are too light to see in legend)
    plt.plot([], [], color='gray', alpha=0.5, linewidth=1, label=f'{num_samples} Sample Paths')

    # Plot percentile bands
    plt.fill_between(x, percentiles['5'], percentiles['95'], color='lightblue', alpha=0.3, label='5th-95th Percentile')
    plt.fill_between(x, percentiles['25'], percentiles['75'], color='blue', alpha=0.3, label='25th-75th Percentile')
    plt.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Forecast')

    # Format plot
    plt.title(f'Forward Return Forecast: {forecast_period} trading days\n({dates[-1]} to ~{forecast_end_date})',
              fontsize=14)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Add key metrics as text in the upper right corner
    info_text = (f"Median Forecast: {final_50th:.2f}%\n"
                 f"5th-95th Range: {final_5th:.2f}% to {final_95th:.2f}%\n"
                 f"25th-75th Range: {final_25th:.2f}% to {final_75th:.2f}%\n"
                 f"Expected Max Drawdown: {np.mean(max_drawdowns):.2f}%")

    if cagr_values:
        info_text += f"\n\nMedian CAGR: {cagr_median:.2f}%\n5th-95th CAGR: {cagr_5th:.2f}% to {cagr_95th:.2f}%"

    plt.text(0.98, 0.98, info_text, transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', horizontalalignment='right')

    # Save the figure
    save_path = os.path.join(forecast_dir, f"{portfolio_name}_forecast_path_{forecast_period}d.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create return distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(final_returns, kde=True, bins=50, color='blue')

    # Add vertical lines for key percentiles
    plt.axvline(x=final_5th, color='red', linestyle='--', alpha=0.7, label=f'5th: {final_5th:.2f}%')
    plt.axvline(x=final_50th, color='green', linestyle='--', linewidth=2, label=f'Median: {final_50th:.2f}%')
    plt.axvline(x=final_95th, color='purple', linestyle='--', alpha=0.7, label=f'95th: {final_95th:.2f}%')

    # Format plot
    plt.title(f'Return Distribution - {forecast_period} Day Forward Forecast', fontsize=14)
    plt.xlabel('Cumulative Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the distribution plot
    dist_plot_path = os.path.join(forecast_dir, f"{portfolio_name}_return_distribution_{forecast_period}d.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create drawdown distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(max_drawdowns, kde=True, bins=50, color='red')

    # Add vertical line for mean and median drawdown
    mean_dd = np.mean(max_drawdowns)
    median_dd = np.median(max_drawdowns)
    dd_95th = np.percentile(max_drawdowns, 95)

    plt.axvline(x=mean_dd, color='black', linestyle='--', linewidth=2,
                label=f'Mean: {mean_dd:.2f}%')
    plt.axvline(x=median_dd, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {median_dd:.2f}%')
    plt.axvline(x=dd_95th, color='purple', linestyle='--', linewidth=2,
                label=f'95th: {dd_95th:.2f}%')

    # Format plot
    plt.title(f'Maximum Drawdown Distribution - {forecast_period} Day Forward Forecast', fontsize=14)
    plt.xlabel('Maximum Drawdown (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the drawdown distribution plot
    dd_plot_path = os.path.join(forecast_dir, f"{portfolio_name}_drawdown_distribution_{forecast_period}d.png")
    plt.savefig(dd_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # If we calculated CAGR, create CAGR distribution plot
    if cagr_values:
        plt.figure(figsize=(10, 6))
        sns.histplot(cagr_values, kde=True, bins=50, color='green')

        # Add vertical lines for key CAGR percentiles
        plt.axvline(x=cagr_5th, color='red', linestyle='--', alpha=0.7, label=f'5th: {cagr_5th:.2f}%')
        plt.axvline(x=cagr_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {cagr_median:.2f}%')
        plt.axvline(x=cagr_95th, color='purple', linestyle='--', alpha=0.7, label=f'95th: {cagr_95th:.2f}%')

        # Format plot
        plt.title(f'CAGR Distribution - {forecast_period} Day Forward Forecast', fontsize=14)
        plt.xlabel('CAGR (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the CAGR distribution plot
        cagr_plot_path = os.path.join(forecast_dir, f"{portfolio_name}_cagr_distribution_{forecast_period}d.png")
        plt.savefig(cagr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Save forecast details to CSV
    forecast_df = pd.DataFrame({
        'Day': range(len(percentiles['50'])),
        'P5': percentiles['5'],
        'P25': percentiles['25'],
        'P50': percentiles['50'],
        'P75': percentiles['75'],
        'P95': percentiles['95']
    })

    forecast_csv_path = os.path.join(forecast_dir, f"{portfolio_name}_forecast_path_{forecast_period}d.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)

    # Save summary statistics to CSV
    stats_dict = {
        'Training_Start_Date': [train_start_date],
        'Training_End_Date': [dates[-1]],
        'Training_Days': [len(train_data)],
        'Forecast_Period_Days': [forecast_period],
        'Approximate_End_Date': [forecast_end_date],
        'Median_Return': [final_50th],
        'Mean_Return': [np.mean(final_returns)],
        'P5_Return': [final_5th],
        'P25_Return': [final_25th],
        'P75_Return': [final_75th],
        'P95_Return': [final_95th],
        'Mean_Max_Drawdown': [mean_dd],
        'Median_Max_Drawdown': [median_dd],
        'P95_Max_Drawdown': [dd_95th]
    }

    # Add CAGR statistics if calculated
    if cagr_values:
        stats_dict.update({
            'Median_CAGR': [cagr_median],
            'Mean_CAGR': [cagr_mean],
            'P5_CAGR': [cagr_5th],
            'P95_CAGR': [cagr_95th]
        })

    stats_df = pd.DataFrame(stats_dict)
    stats_csv_path = os.path.join(forecast_dir, f"{portfolio_name}_forecast_stats_{forecast_period}d.csv")
    stats_df.to_csv(stats_csv_path, index=False)

    # Print summary report
    print("\n" + "=" * 50)
    print(f"FORWARD FORECAST SUMMARY: {portfolio_name}")
    print("=" * 50)
    print(f"Training Period: {train_start_date} to {dates[-1]} ({len(train_data)} trading days)")
    print(f"Forecast Period: {dates[-1]} to ~{forecast_end_date} ({forecast_period} trading days)")
    print("\nForecast Return:")
    print(f"  Median: {final_50th:.2f}%")
    print(f"  25th-75th Range: {final_25th:.2f}% to {final_75th:.2f}%")
    print(f"  5th-95th Range: {final_5th:.2f}% to {final_95th:.2f}%")
    print("\nExpected Drawdown:")
    print(f"  Mean: {mean_dd:.2f}%")
    print(f"  Median: {median_dd:.2f}%")
    print(f"  95th Percentile: {dd_95th:.2f}%")

    if cagr_values:
        print("\nForecast CAGR:")
        print(f"  Median: {cagr_median:.2f}%")
        print(f"  5th-95th Range: {cagr_5th:.2f}% to {cagr_95th:.2f}%")

    print("\nPosition Sizing Recommendation:")
    if final_50th > 25:
        position_text = "AGGRESSIVE: Expected returns significantly exceed historical average"
    elif final_50th > 15:
        position_text = "MODERATE-AGGRESSIVE: Expected returns exceed historical average"
    elif final_50th > 5:
        position_text = "MODERATE: Expected returns in line with historical average"
    else:
        position_text = "CONSERVATIVE: Expected returns below historical average"

    # Adjust based on drawdown risk
    if dd_95th > 10:
        position_text += " (consider reducing due to elevated drawdown risk)"
    elif dd_95th < 3:
        position_text += " (could consider increasing due to low drawdown risk)"

    print(f"  {position_text}")

    print("\nFiles saved:")
    print(f"  Forecast path: {save_path}")
    print(f"  Return distribution: {dist_plot_path}")
    print(f"  Drawdown distribution: {dd_plot_path}")
    if cagr_values:
        print(f"  CAGR distribution: {cagr_plot_path}")
    print(f"  Forecast path data: {forecast_csv_path}")
    print(f"  Summary statistics: {stats_csv_path}")

    return {
        'median_return': final_50th,
        'percentile_5': final_5th,
        'percentile_95': final_95th,
        'mean_drawdown': mean_dd,
        'median_drawdown': median_dd,
        'drawdown_95th': dd_95th,
        'cagr_median': cagr_median if cagr_values else None
    }


def main():
    """
    Main function to run the Monte Carlo analysis on a Composer portfolio
    """
    # Output directory
    output_dir = "composer_monte_carlo_results"
    os.makedirs(output_dir, exist_ok=True)

    # Default composer URL if none is provided
    default_url = 'https://app.composer.trade/symphony/BrnnCuy0Dhz3DjaAZbFt/details'

    # Check if user wants to load existing data
    use_existing = input("Do you want to use previously saved returns data? (y/n, default: n): ").lower() == 'y'

    if use_existing:
        # List available saved returns files
        saved_files = [f for f in os.listdir(output_dir) if f.endswith('_daily_returns.csv')]

        if not saved_files:
            print("No saved returns data found. Will fetch new data.")
            use_existing = False
        else:
            print("\nAvailable saved returns:")
            for i, file in enumerate(saved_files, 1):
                name = file.replace('_daily_returns.csv', '').replace('_', ' ')
                print(f"{i}. {name}")

            try:
                choice = int(input("\nSelect a file to load (number), or 0 to fetch new data: "))
                if choice == 0:
                    use_existing = False
                elif 1 <= choice <= len(saved_files):
                    selected_file = saved_files[choice - 1]
                    clean_symphony_name = selected_file.replace('_daily_returns.csv', '')
                    returns_path = os.path.join(output_dir, selected_file)

                    # Load the saved returns data
                    print(f"Loading returns data from {returns_path}")
                    returns_df = pd.read_csv(returns_path)
                    date_strs = returns_df['Date'].tolist()
                    returns = returns_df['Daily_Return'].tolist()

                    # Extract symphony name for display
                    symphony_name = clean_symphony_name.replace('_', ' ')
                    print(f"\nLoaded portfolio: {symphony_name}")
                    print(f"Historical data period: {date_strs[0]} to {date_strs[-1]}")
                    print(f"Total trading days: {len(returns)}")
                else:
                    print("Invalid choice. Will fetch new data.")
                    use_existing = False
            except (ValueError, IndexError):
                print("Invalid input. Will fetch new data.")
                use_existing = False

    if not use_existing:
        # Get input from user
        symphony_url = input(f"Enter Composer Symphony URL (default: {default_url}): ") or default_url

        # Set date range
        today = date.today().strftime('%Y-%m-%d')
        start_date = input(f"Enter start date (YYYY-MM-DD), or leave blank for all available data: ") or '2000-01-01'
        end_date = input(f"Enter end date (YYYY-MM-DD), or leave blank for today ({today}): ") or today

        # Fetch backtest data from Composer
        print(f"Fetching data from Composer: {symphony_url}")
        try:
            allocations_df, symphony_name, tickers = fetch_backtest(symphony_url, start_date, end_date)
        except Exception as e:
            print(f"Error fetching data from Composer: {str(e)}")
            return

        # Clean symphony name for file naming (remove special characters that could cause file system issues)
        clean_symphony_name = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in symphony_name)
        clean_symphony_name = clean_symphony_name.replace(' ', '_')

        # Calculate daily returns
        print(f"Calculating daily returns for {symphony_name}...")
        daily_returns, dates = calculate_portfolio_returns(allocations_df, tickers)

        # Convert dates to strings for easier handling
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]

        # Check if we have enough data
        if len(daily_returns) < 60:  # Need at least 60 days of data
            print("Error: Not enough historical data for Monte Carlo analysis (minimum 60 days required).")
            return

        # Make sure date_strs and daily_returns have the same length
        if len(date_strs) != len(daily_returns):
            print(f"Warning: Length mismatch - date_strs: {len(date_strs)}, daily_returns: {len(daily_returns)}")
            # Trim to the shorter length
            min_length = min(len(date_strs), len(daily_returns))
            date_strs = date_strs[:min_length]
            # For daily_returns, we need to handle it as a pandas Series
            if isinstance(daily_returns, pd.Series):
                daily_returns = daily_returns.iloc[:min_length]
            else:
                daily_returns = daily_returns[:min_length]

        # Create the DataFrame
        returns_df = pd.DataFrame({'Date': date_strs, 'Daily_Return': daily_returns})

        # Extract the daily returns as a simple list for Monte Carlo simulation
        returns = returns_df['Daily_Return'].tolist()

        # Save the daily returns data before running the tests with symphony name prefixed
        returns_path = os.path.join(output_dir, f"{clean_symphony_name}_daily_returns.csv")
        returns_df.to_csv(returns_path, index=False)
        print(f"Daily returns saved to: {returns_path}")

        print(f"\nAnalyzing portfolio: {symphony_name}")
        print(f"Historical data period: {date_strs[0]} to {date_strs[-1]}")
        print(f"Total trading days: {len(returns)}")

    # Define simulation parameters
    annual_periods = 252

    # Set simulation lengths for different time periods
    simulation_length_3m = int(annual_periods / 4)  # 3 months (1 quarter)
    simulation_length_6m = int(annual_periods / 2)  # 6 months
    simulation_length_1y = annual_periods  # 1 year
    simulation_length_2y = annual_periods * 2  # 2 years

    # Main analysis loop
    while True:
        # Ask user which test to run
        print("\nAvailable test modes:")
        print("1. Standard Walk-Forward (train on all data except last N days, test on last N days)")
        print("2. Rolling Walk (start at beginning, train on x days, simulate y forward days)")
        print("3. Expanding Window (start with initial window, increase by 1 year each iteration)")
        print("4. All Historical Tests (modes 1-3)")
        print("5. Forward Forecast (generate future forecast from most recent data)")
        print("6. Complete Suite (all modes 1-5)")
        print("0. Exit")

        mode = input("Enter test mode (0-6, default: 1): ") or "1"

        # Check for exit condition
        if mode == "0":
            print("\nExiting Monte Carlo analysis. Results have been saved to:", output_dir)
            break

        # Run standard walk-forward tests if selected
        if mode in ["1", "4", "6"]:
            print("\n------------------------------------------")
            print("Standard Walk-Forward Tests")
            print("------------------------------------------")

            # Define walk-forward test periods
            walk_forward_test_periods = [
                simulation_length_3m,  # ~3 months
                simulation_length_6m,  # ~6 months
                simulation_length_1y,  # ~1 year
            ]

            # Add 2-year test if we have enough data
            if len(returns) >= (simulation_length_2y + 60):  # Need at least 60 days of training data
                walk_forward_test_periods.append(simulation_length_2y)

            walk_forward_results = []

            for period_length in walk_forward_test_periods:
                if period_length <= len(returns):
                    try:
                        # Added clean_symphony_name to the function call
                        result = run_walk_forward_test(date_strs, returns, period_length, output_dir,
                                                       clean_symphony_name)
                        if result:
                            walk_forward_results.append(result)
                    except Exception as e:
                        print(f"Error in walk-forward test for {period_length} days: {e}")
                else:
                    print(f"Skipping {period_length}-day test: not enough historical data")

            # Summarize walk-forward test results
            if walk_forward_results:
                print("\nWalk-Forward Test Summary:")
                print(
                    f"{'Period':<10} {'Start Date':<12} {'End Date':<12} {'Actual Return':<15} {'Forecast':<15} {'Error %':<10} {'Percentile':<10} {'DD Calendar':<12} {'In 90% CI':<10}")
                print("-" * 110)

                for result in walk_forward_results:
                    period_desc = f"{result['period_length']}d"
                    in_90_ci = "Yes" if result['in_90_interval'] else "No"
                    in_50_ci = "Yes" if result['in_50_interval'] else "No"

                    # Format percentages with the % symbol after the number
                    actual_return_fmt = f"{result['actual_final_return']:.2f}%"
                    median_forecast_fmt = f"{result['median_forecast']:.2f}%"
                    percent_error_fmt = f"{result['percent_error']:.2f}%"
                    percentile_fmt = f"{result['actual_percentile']:.1f}%"
                    dd_calendar = f"{result['actual_dd_duration_calendar']} days"

                    print(f"{period_desc:<10} {result['test_start_date']:<12} {result['test_end_date']:<12} "
                          f"{actual_return_fmt:<15} {median_forecast_fmt:<15} {percent_error_fmt:<10} {percentile_fmt:<10} "
                          f"{dd_calendar:<12} {in_90_ci:<10}")

                # Create a DataFrame for walk-forward test results
                wf_results_df = pd.DataFrame(walk_forward_results)

                # Round all numeric columns to 2 decimal places
                for column in wf_results_df.columns:
                    if wf_results_df[column].dtype in ['float64', 'float32']:
                        wf_results_df[column] = wf_results_df[column].round(2)

                # Save walk-forward test results to CSV with portfolio name prefixed
                wf_csv_path = os.path.join(output_dir, f"{clean_symphony_name}_walk_forward_results.csv")
                wf_results_df.to_csv(wf_csv_path, index=False)
                print(f"\nWalk-forward test results saved to CSV: {wf_csv_path}")

                # Create a chart comparing actual vs forecast returns
                plt.figure(figsize=(10, 6))
                periods = [f"{r['period_length']}d" for r in walk_forward_results]
                actual_returns = [r['actual_final_return'] for r in walk_forward_results]
                forecast_returns = [r['median_forecast'] for r in walk_forward_results]

                x = range(len(periods))
                width = 0.35

                plt.bar([i - width / 2 for i in x], actual_returns, width, label='Actual Return', color='green')
                plt.bar([i + width / 2 for i in x], forecast_returns, width, label='Forecast Return', color='blue')

                plt.xlabel('Time Period')
                plt.ylabel('Cumulative Return (%)')
                plt.title(f'Actual vs Forecast Returns by Time Period - {symphony_name}')
                plt.xticks(x, periods)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Modified save path with portfolio name prefixed
                comparison_chart_path = os.path.join(output_dir, f"{clean_symphony_name}_comparison.png")
                plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Comparison chart saved to: {comparison_chart_path}")

        # Run rolling walk tests if selected
        if mode in ["2", "4", "6"]:
            print("\n------------------------------------------")
            print("Rolling Walk-Forward Tests")
            print("------------------------------------------")

            # Ask which type of rolling walk test to run
            print("\nRolling Walk-Forward Test Options:")
            print("1. Standard sliding window (train on N days, test on M days)")
            print("2. Fixed training window (train on specific date range, test on M days)")

            rolling_mode = input("Enter rolling test mode (1 or 2, default: 1): ") or "1"

            # Ask if test windows should be allowed to overlap with training windows
            allow_overlap = input(
                "Allow test windows to start from the beginning of the historical record? (y/n, default: n): ").lower() == 'y'

            if allow_overlap:
                print("Warning: Test windows will be allowed to overlap with training windows")
                print("This means some test data may have been included in the training data")

            if rolling_mode == "1":  # Standard sliding window
                # Get parameters for standard rolling walk test
                default_train_length = 252  # Default to 1 year training
                default_test_length = 252  # Default to 1 year testing
                default_step_size = 63  # Default to quarterly steps (3 months)

                train_length = input(f"Enter training period length in days (default: {default_train_length}): ")
                train_length = int(train_length) if train_length else default_train_length

                test_length = input(f"Enter test period length in days (default: {default_test_length}): ")
                test_length = int(test_length) if test_length else default_test_length

                step_size = input(f"Enter step size in days (default: {default_step_size}): ")
                step_size = int(step_size) if step_size else default_step_size

                extend_last = input("Extend last iteration to current date? (y/n, default: y): ").lower() != 'n'

                # Check if we have enough data
                if len(returns) < (train_length + test_length) and not allow_overlap:
                    print(
                        f"Error: Not enough data for non-overlapping rolling walk test. Need at least {train_length + test_length} days.")
                    print("Consider allowing overlap or using a shorter training/test period.")
                else:
                    try:
                        # Run the rolling walk-forward test with sliding window
                        run_rolling_walk_forward_test(
                            date_strs,
                            returns,
                            train_period_length=train_length,
                            test_period_length=test_length,
                            output_dir=output_dir,
                            portfolio_name=clean_symphony_name,
                            step_size=step_size,
                            fixed_train_start_date=None,
                            fixed_train_end_date=None,
                            extend_last_to_current=extend_last,
                            allow_overlap=allow_overlap
                        )
                    except Exception as e:
                        print(f"Error in rolling walk-forward test: {e}")
            else:  # Fixed training window
                # Get fixed date range for training window
                print(f"\nAvailable date range: {date_strs[0]} to {date_strs[-1]}")

                default_train_start = date_strs[0]
                # Set default training end date to ~halfway through the available data
                middle_idx = len(date_strs) // 2
                default_train_end = date_strs[middle_idx]

                train_start_date = input(f"Enter training window start date (default: {default_train_start}): ")
                train_start_date = train_start_date if train_start_date else default_train_start

                train_end_date = input(f"Enter training window end date (default: {default_train_end}): ")
                train_end_date = train_end_date if train_end_date else default_train_end

                default_test_length = 252  # Default to 1 year testing
                default_step_size = 63  # Default to quarterly steps (3 months)

                test_length = input(f"Enter test period length in days (default: {default_test_length}): ")
                test_length = int(test_length) if test_length else default_test_length

                step_size = input(f"Enter step size in days (default: {default_step_size}): ")
                step_size = int(step_size) if step_size else default_step_size

                extend_last = input("Extend last iteration to current date? (y/n, default: y): ").lower() != 'n'

                try:
                    # Run the rolling walk-forward test with fixed training window
                    run_rolling_walk_forward_test(
                        date_strs,
                        returns,
                        train_period_length=None,  # Not used in fixed window mode
                        test_period_length=test_length,
                        output_dir=output_dir,
                        portfolio_name=clean_symphony_name,
                        step_size=step_size,
                        fixed_train_start_date=train_start_date,
                        fixed_train_end_date=train_end_date,
                        extend_last_to_current=extend_last,
                        allow_overlap=allow_overlap
                    )
                except Exception as e:
                    print(f"Error in fixed window rolling walk-forward test: {e}")

        # Run expanding window test if selected
        if mode in ["3", "4", "6"]:
            print("\n------------------------------------------")
            print("Expanding Window Tests")
            print("------------------------------------------")

            # Get parameters for expanding window test
            default_initial_train = 252  # Default to 1 year initial training
            default_test_length = 252  # Default to 1 year testing
            default_expansion_size = 252  # Default to 1 year expansion

            initial_train = input(
                f"Enter initial training period length in days (default: {default_initial_train}): ")
            initial_train = int(initial_train) if initial_train else default_initial_train

            test_length = input(f"Enter test period length in days (default: {default_test_length}): ")
            test_length = int(test_length) if test_length else default_test_length

            expansion_size = input(f"Enter window expansion size in days (default: {default_expansion_size}): ")
            expansion_size = int(expansion_size) if expansion_size else default_expansion_size

            # Check if we have enough data
            if len(returns) < (initial_train + test_length):
                print(
                    f"Error: Not enough data for expanding window test. Need at least {initial_train + test_length} days.")
            else:
                try:
                    # Run the expanding window test
                    run_expanding_window_test(
                        date_strs,
                        returns,
                        initial_train_period=initial_train,
                        test_period_length=test_length,
                        expansion_size=expansion_size,
                        output_dir=output_dir,
                        portfolio_name=clean_symphony_name
                    )
                except Exception as e:
                    print(f"Error in expanding window test: {e}")

        # Add the Forward Forecast mode
        if mode in ["5", "6"]:
            try:
                forecast_results = add_forward_forecast_mode(date_strs, returns, output_dir, clean_symphony_name)
            except Exception as e:
                print(f"Error in forward forecast: {e}")
                import traceback
                traceback.print_exc()

        # Display completion message for the current analysis
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print("You can run another analysis or exit.")


if __name__ == "__main__":
    main()
