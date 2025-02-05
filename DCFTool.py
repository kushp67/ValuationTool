import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

sns.set_theme(style="whitegrid", palette="muted")

# ----- Custom CSS for a Modern, Aesthetic Look -----
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #34495e;
        font-weight: 600;
    }
    .input-dashboard {
        background: #ffffff;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.6em 1.2em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        margin: 10px 5px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .card h3 {
        color: #34495e;
        margin-bottom: 5px;
    }
    .card h1 {
        font-size: 2rem;
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----- Helper Display Functions -----

def display_metric_card(title, value, unit="$", value_color="#27ae60"):
    card_html = f"""
    <div class="card">
      <h3>{title}</h3>
      <h1 style="color: {value_color};">{unit}{value:,.2f}</h1>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def display_metric_cards(metrics_dict, cols=3):
    keys = list(metrics_dict.keys())
    for i in range(0, len(keys), cols):
        cols_container = st.columns(cols)
        for j, key in enumerate(keys[i:i+cols]):
            value = metrics_dict[key]
            color = "#27ae60" if j % 2 == 0 else "#2980b9"
            cols_container[j].markdown(f"**{key}:**")
            cols_container[j].markdown(f"<div class='card'><h1 style='color: {color};'>{value:,.2f}</h1></div>", unsafe_allow_html=True)

def display_ev_comparison(calculated_ev, market_cap):
    delta = calculated_ev - market_cap
    delta_color = "#27ae60" if delta >= 0 else "#e74c3c"
    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric_card("DCF EV", calculated_ev)
    with col2:
        display_metric_card("Market Cap", market_cap)
    with col3:
        display_metric_card("Difference", delta, unit="", value_color=delta_color)

# ----- Data Retrieval Functions -----

def fetch_latest_fcf(ticker):
    stock = yf.Ticker(ticker)
    try:
        latest_fcf = stock.cashflow.loc['Free Cash Flow'].iloc[0]
    except Exception as e:
        st.error(f"Error retrieving FCF for {ticker}: {e}")
        return None
    return latest_fcf

def fetch_additional_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    metrics = {
        "Market Cap": info.get("marketCap", "N/A"),
        "Beta": info.get("beta", "N/A"),
        "Price/Earnings": info.get("trailingPE", "N/A"),
        "Revenue": info.get("totalRevenue", "N/A"),
        "Profit Margin": info.get("profitMargins", "N/A"),
        "Dividend Rate": info.get("dividendRate", None),
        "Dividend Yield": info.get("dividendYield", None),
        "Price-To-Book": info.get("priceToBook", None)
    }
    return metrics

def fetch_financials(ticker):
    stock = yf.Ticker(ticker)
    try:
        fin = stock.financials
    except Exception as e:
        st.error(f"Error retrieving financials for {ticker}: {e}")
        return None
    return fin

def fetch_balance_sheet(ticker):
    stock = yf.Ticker(ticker)
    try:
        bs = stock.balance_sheet
    except Exception as e:
        st.error(f"Error retrieving balance sheet for {ticker}: {e}")
        return None
    return bs

def fetch_historical_prices(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period=period)
    except Exception as e:
        st.error(f"Error retrieving historical prices for {ticker}: {e}")
        return None
    return hist

# ----- Historical FCF Values -----

def get_last_five_fcf(ticker):
    stock = yf.Ticker(ticker)
    if hasattr(stock, "quarterly_cashflow"):
        cf = stock.quarterly_cashflow
    else:
        cf = stock.cashflow
    if 'Free Cash Flow' not in cf.index:
        return None
    try:
        cf.columns = pd.to_datetime(cf.columns, errors='coerce')
        cf = cf.sort_index(axis=1)
    except Exception:
        pass
    fcf_series = cf.loc['Free Cash Flow'].dropna()
    if len(fcf_series) >= 5:
        last_values = fcf_series.iloc[-5:]
    else:
        last_values = fcf_series
    df = pd.DataFrame(last_values)
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.rename(columns={"Free Cash Flow": "FCF"})
    return df

# ----- Calculation Functions -----

def simple_dcf(latest_fcf, discount_rate=0.1, forecast_years=5, fcf_growth=0.05, 
               terminal_method="Gordon Growth", terminal_growth=0.02, exit_multiple=10):
    projected_fcfs = [latest_fcf * ((1 + fcf_growth) ** year) for year in range(1, forecast_years + 1)]
    pv_fcfs = [fcf / ((1 + discount_rate) ** year) for year, fcf in enumerate(projected_fcfs, start=1)]
    last_fcf = projected_fcfs[-1]
    if terminal_method == "Gordon Growth":
        if discount_rate <= terminal_growth:
            raise ValueError("Discount rate must be greater than terminal growth rate.")
        terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    elif terminal_method == "Exit Multiple":
        terminal_value = last_fcf * exit_multiple
    else:
        raise ValueError("Invalid terminal method.")
    pv_terminal = terminal_value / ((1 + discount_rate) ** forecast_years)
    enterprise_value = sum(pv_fcfs) + pv_terminal
    return enterprise_value, projected_fcfs, pv_fcfs, terminal_value, pv_terminal

def calculate_sensitivity(latest_fcf, forecast_years, fcf_growth, terminal_method, exit_multiple,
                          base_discount_rate, base_terminal_growth, discount_range, growth_range):
    discount_rates = np.linspace(base_discount_rate - discount_range, base_discount_rate + discount_range, 10)
    terminal_growth_rates = np.linspace(base_terminal_growth - growth_range, base_terminal_growth + growth_range, 10)
    results = []
    for dr in discount_rates:
        row = []
        for tg in terminal_growth_rates:
            try:
                ev, _, _, _, _ = simple_dcf(latest_fcf,
                                            discount_rate=dr,
                                            forecast_years=forecast_years,
                                            fcf_growth=fcf_growth,
                                            terminal_method=terminal_method,
                                            terminal_growth=tg,
                                            exit_multiple=exit_multiple)
                row.append(ev)
            except Exception:
                row.append(np.nan)
        results.append(row)
    sensitivity_df = pd.DataFrame(results, index=np.round(discount_rates, 3), columns=np.round(terminal_growth_rates, 3))
    return sensitivity_df

def monte_carlo_dcf(latest_fcf, base_discount_rate, forecast_years, base_fcf_growth,
                    terminal_method, base_terminal_growth, exit_multiple,
                    iterations=1000, discount_rate_std=0.005, fcf_growth_std=0.005, terminal_growth_std=0.002):
    evs = []
    for _ in range(iterations):
        dr_sample = np.random.normal(base_discount_rate, discount_rate_std)
        fcf_growth_sample = np.random.normal(base_fcf_growth, fcf_growth_std)
        tg_sample = np.random.normal(base_terminal_growth, terminal_growth_std)
        try:
            ev, _, _, _, _ = simple_dcf(latest_fcf,
                                        discount_rate=dr_sample,
                                        forecast_years=forecast_years,
                                        fcf_growth=fcf_growth_sample,
                                        terminal_method=terminal_method,
                                        terminal_growth=tg_sample,
                                        exit_multiple=exit_multiple)
            evs.append(ev)
        except Exception:
            continue
    return evs

# ----- Additional Valuation Modules -----

def multiples_analysis(ticker):
    metrics = fetch_additional_metrics(ticker)
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get("marketCap", None)
    total_debt = info.get("totalDebt", 0)
    total_cash = info.get("totalCash", 0)
    if market_cap is not None:
        ev = market_cap + total_debt - total_cash
    else:
        ev = None
    fin = fetch_financials(ticker)
    if fin is not None and 'Ebitda' in fin.index:
        ebitda = fin.loc['Ebitda'].iloc[0]
    else:
        ebitda = None
    multiples = {}
    multiples["P/E"] = info.get("trailingPE", None)
    multiples["Price-to-Book"] = info.get("priceToBook", None)
    if ev and ebitda and ebitda != 0:
        multiples["EV/EBITDA"] = ev / ebitda
    else:
        multiples["EV/EBITDA"] = None
    multiples["EV/Sales"] = info.get("enterpriseToRevenue", None)
    return multiples

def ddm_valuation(ticker, required_return, dividend_growth):
    stock = yf.Ticker(ticker)
    info = stock.info
    dividend_rate = info.get("dividendRate", None)
    if dividend_rate is None or dividend_rate == 0 or required_return <= dividend_growth:
        return None
    D1 = dividend_rate
    intrinsic_value = D1 * (1 + dividend_growth) / (required_return - dividend_growth)
    return intrinsic_value

def epv_valuation(ticker, discount_rate):
    fin = fetch_financials(ticker)
    if fin is None or 'Net Income' not in fin.index:
        return None
    net_income = fin.loc['Net Income'].iloc[0]
    return net_income / discount_rate

def nav_valuation(ticker):
    bs = fetch_balance_sheet(ticker)
    if bs is None:
        return None
    if "Total Assets" in bs.index:
        total_assets = bs.loc["Total Assets"].iloc[0]
    elif "totalAssets" in bs.index:
        total_assets = bs.loc["totalAssets"].iloc[0]
    else:
        return None
    if "Total Liabilities Net Minority Interest" in bs.index:
        total_liab = bs.loc["Total Liabilities Net Minority Interest"].iloc[0]
    elif "Total Liab" in bs.index:
        total_liab = bs.loc["Total Liab"].iloc[0]
    elif "totalLiab" in bs.index:
        total_liab = bs.loc["totalLiab"].iloc[0]
    else:
        return None
    return total_assets - total_liab

def balance_sheet_analysis(ticker):
    bs = fetch_balance_sheet(ticker)
    if bs is None:
        return None
    analysis = {}
    # Total Assets & Liabilities
    if "Total Assets" in bs.index:
        analysis["Total Assets"] = bs.loc["Total Assets"].iloc[0]
    elif "totalAssets" in bs.index:
        analysis["Total Assets"] = bs.loc["totalAssets"].iloc[0]
    if "Total Liabilities Net Minority Interest" in bs.index:
        analysis["Total Liabilities"] = bs.loc["Total Liabilities Net Minority Interest"].iloc[0]
    elif "Total Liab" in bs.index:
        analysis["Total Liabilities"] = bs.loc["Total Liab"].iloc[0]
    elif "totalLiab" in bs.index:
        analysis["Total Liabilities"] = bs.loc["totalLiab"].iloc[0]
    # Equity
    if "Stockholders Equity" in bs.index:
        analysis["Stockholders Equity"] = bs.loc["Stockholders Equity"].iloc[0]
    elif "Common Stock Equity" in bs.index:
        analysis["Stockholders Equity"] = bs.loc["Common Stock Equity"].iloc[0]
    # Tangible Book Value
    if "Tangible Book Value" in bs.index:
        analysis["Tangible Book Value"] = bs.loc["Tangible Book Value"].iloc[0]
    # Current Assets & Liabilities
    if "Current Assets" in bs.index:
        analysis["Current Assets"] = bs.loc["Current Assets"].iloc[0]
    if "Current Liabilities" in bs.index:
        analysis["Current Liabilities"] = bs.loc["Current Liabilities"].iloc[0]
    if "Current Assets" in analysis and "Current Liabilities" in analysis:
        analysis["Working Capital"] = analysis["Current Assets"] - analysis["Current Liabilities"]
        if analysis["Current Liabilities"] != 0:
            analysis["Current Ratio"] = analysis["Current Assets"] / analysis["Current Liabilities"]
        else:
            analysis["Current Ratio"] = None
    # Total Debt
    if "Total Debt" in bs.index:
        analysis["Total Debt"] = bs.loc["Total Debt"].iloc[0]
    elif "totalDebt" in bs.index:
        analysis["Total Debt"] = bs.loc["totalDebt"].iloc[0]
    # Net Debt
    if "Net Debt" in bs.index:
        analysis["Net Debt"] = bs.loc["Net Debt"].iloc[0]
    else:
        if "Cash And Cash Equivalents And Short Term Investments" in bs.index and "Total Debt" in analysis:
            cash = bs.loc["Cash And Cash Equivalents And Short Term Investments"].iloc[0]
            analysis["Net Debt"] = analysis["Total Debt"] - cash
    # Leverage ratios
    if "Total Debt" in analysis and "Stockholders Equity" in analysis and analysis["Stockholders Equity"] != 0:
        analysis["Debt-to-Equity Ratio"] = analysis["Total Debt"] / analysis["Stockholders Equity"]
    else:
        analysis["Debt-to-Equity Ratio"] = None
    if "Total Liabilities" in analysis and "Total Assets" in analysis and analysis["Total Assets"] != 0:
        analysis["Debt-to-Asset Ratio"] = analysis["Total Liabilities"] / analysis["Total Assets"]
    else:
        analysis["Debt-to-Asset Ratio"] = None
    if "Stockholders Equity" in analysis and "Total Assets" in analysis and analysis["Total Assets"] != 0:
        analysis["Equity Ratio"] = analysis["Stockholders Equity"] / analysis["Total Assets"]
    else:
        analysis["Equity Ratio"] = None
    # Per share metrics
    stock = yf.Ticker(ticker)
    info = stock.info
    shares = info.get("sharesOutstanding", None)
    if shares:
        analysis["Shares Outstanding"] = shares
        if "Stockholders Equity" in analysis and analysis["Stockholders Equity"] is not None:
            analysis["Book Value Per Share"] = analysis["Stockholders Equity"] / shares
        if "Tangible Book Value" in analysis and analysis["Tangible Book Value"] is not None:
            analysis["Tangible Book Value Per Share"] = analysis["Tangible Book Value"] / shares
    return analysis

def eva_calculator(ticker, discount_rate, tax_rate):
    fin = fetch_financials(ticker)
    if fin is None or "Ebit" not in fin.index:
        return None
    EBIT = fin.loc["Ebit"].iloc[0]
    NOPAT = EBIT * (1 - tax_rate)
    bs = fetch_balance_sheet(ticker)
    if bs is None:
        return None
    if "Stockholders Equity" in bs.index:
        equity = bs.loc["Stockholders Equity"].iloc[0]
    elif "Common Stock Equity" in bs.index:
        equity = bs.loc["Common Stock Equity"].iloc[0]
    else:
        return None
    if "Total Debt" in bs.index:
        total_debt = bs.loc["Total Debt"].iloc[0]
    elif "totalDebt" in bs.index:
        total_debt = bs.loc["totalDebt"].iloc[0]
    else:
        total_debt = 0
    if "Cash And Cash Equivalents And Short Term Investments" in bs.index:
        cash = bs.loc["Cash And Cash Equivalents And Short Term Investments"].iloc[0]
    else:
        cash = 0
    invested_capital = equity + total_debt - cash
    eva = NOPAT - (invested_capital * discount_rate)
    return eva, NOPAT, invested_capital

def roic_analysis(ticker, discount_rate, tax_rate):
    fin = fetch_financials(ticker)
    if fin is None or "Ebit" not in fin.index:
        return None
    EBIT = fin.loc["Ebit"].iloc[0]
    NOPAT = EBIT * (1 - tax_rate)
    bs = fetch_balance_sheet(ticker)
    if bs is None:
        return None
    if "Stockholders Equity" in bs.index:
        equity = bs.loc["Stockholders Equity"].iloc[0]
    elif "Common Stock Equity" in bs.index:
        equity = bs.loc["Common Stock Equity"].iloc[0]
    else:
        return None
    if "Total Debt" in bs.index:
        total_debt = bs.loc["Total Debt"].iloc[0]
    elif "totalDebt" in bs.index:
        total_debt = bs.loc["totalDebt"].iloc[0]
    else:
        total_debt = 0
    if "Cash And Cash Equivalents And Short Term Investments" in bs.index:
        cash = bs.loc["Cash And Cash Equivalents And Short Term Investments"].iloc[0]
    else:
        cash = 0
    invested_capital = equity + total_debt - cash
    if invested_capital == 0:
        return None
    ROIC = NOPAT / invested_capital
    return ROIC, discount_rate

def historical_trends_analysis(ticker):
    stock = yf.Ticker(ticker)
    fin = stock.financials
    trends = {}
    if fin is not None:
        if "Total Revenue" in fin.index:
            trends["Total Revenue"] = fin.loc["Total Revenue"]
        if "Net Income" in fin.index:
            trends["Net Income"] = fin.loc["Net Income"]
    cf = stock.cashflow
    if cf is not None and "Free Cash Flow" in cf.index:
        trends["Free Cash Flow"] = cf.loc["Free Cash Flow"]
    df_list = []
    for key, series in trends.items():
        # Convert series to a DataFrame with one column named as the metric
        temp = series.to_frame(name=key)
        df_list.append(temp)
    if df_list:
        combined = pd.concat(df_list, axis=1)
        combined.index = pd.to_datetime(combined.index).strftime("%Y-%m-%d")
        return combined
    else:
        return None

# ----- Utility Functions for PDF Export -----

def to_csv(df):
    return df.to_csv(index=False, float_format="%.2f").encode('utf-8')

def generate_pdf_report(report_data, chart_images):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Valuation Report for {report_data['ticker']}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"DCF Value: ${report_data.get('DCF', 'N/A')}", ln=True)
    if report_data.get("DDM"):
        pdf.cell(0, 10, f"DDM Value: ${report_data['DDM']:,.2f}", ln=True)
    if report_data.get("EPV"):
        pdf.cell(0, 10, f"EPV: ${report_data['EPV']:,.2f}", ln=True)
    if report_data.get("NAV"):
        pdf.cell(0, 10, f"NAV: ${report_data['NAV']:,.2f}", ln=True)
    if report_data.get("Multiples"):
        pdf.cell(0, 10, "Multiples Analysis:", ln=True)
        for key, value in report_data["Multiples"].items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)
    if report_data.get("BalanceSheet"):
        pdf.cell(0, 10, "Balance Sheet Analysis:", ln=True)
        for key, value in report_data["BalanceSheet"].items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)
    if report_data.get("PeerComparison") is not None:
        pdf.cell(0, 10, "Peer Comparison:", ln=True)
        for idx, row in report_data["PeerComparison"].iterrows():
            pdf.cell(0, 10, f"{row['Ticker']}: P/E {row['P/E']}, EV/EBITDA {row['EV/EBITDA']}", ln=True)
    if report_data.get("EVA"):
        pdf.cell(0, 10, f"EVA: ${report_data['EVA']:,.2f}", ln=True)
    if report_data.get("ROIC"):
        pdf.cell(0, 10, f"ROIC: {report_data['ROIC']*100:.2f}%", ln=True)
    if report_data.get("Market Cap"):
        pdf.cell(0, 10, f"Market Cap: ${report_data['Market Cap']:,.2f}", ln=True)
    if report_data.get("Difference"):
        pdf.cell(0, 10, f"Difference (DCF EV - Market Cap): ${report_data['Difference']:,.2f}", ln=True)
    pdf.ln(10)
    # Embed charts by writing temporary files (FPDF needs file paths)
    import tempfile
    for title, img in chart_images.items():
        pdf.cell(0, 10, title, ln=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmpfile.write(img.getvalue())
            tmpfile.flush()
            pdf.image(tmpfile.name, w=pdf.w - 40)
            pdf.ln(10)
    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output

# ----- Top Dashboard for Inputs -----
with st.container():
    st.markdown("<div class='input-dashboard'><h2>Valuation App Inputs</h2></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    with col2:
        discount_rate = st.number_input("Discount Rate (WACC)", min_value=0.0, max_value=1.0, value=0.10, step=0.005)
    with col3:
        forecast_years = st.number_input("Forecast Period (Years)", min_value=1, max_value=20, value=5, step=1)
    with col4:
        manual_fcf_growth = st.number_input("Manual FCF Growth Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.005)
    
    col5, col6 = st.columns(2)
    with col5:
        terminal_method = st.selectbox("Terminal Value Method", ["Gordon Growth", "Exit Multiple"])
    with col6:
        if terminal_method == "Gordon Growth":
            terminal_growth = st.number_input("Terminal Growth Rate", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
            exit_multiple = None
        else:
            exit_multiple = st.number_input("Exit Multiple", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
            terminal_growth = None
    
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        sensitivity_enabled = st.checkbox("Show Sensitivity Analysis", value=False)
    with col8:
        discount_range = st.number_input("Discount Rate Variation", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
    with col9:
        growth_range = st.number_input("Terminal Growth Variation", min_value=0.0, max_value=0.05, value=0.005, step=0.001)
    with col10:
        monte_carlo_enabled = st.checkbox("Run Monte Carlo Simulation", value=False)
    
    # Peer Comparison input
    peer_input = st.text_input("Peer Tickers (comma separated)", value="MSFT,GOOGL,AMZN")
    # EVA and ROIC tax rate input
    tax_rate = st.number_input("Effective Tax Rate (for EVA/ROIC)", min_value=0.0, max_value=1.0, value=0.21, step=0.01)
    
    if monte_carlo_enabled:
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        with col_mc1:
            iterations = st.number_input("Iterations", min_value=100, max_value=5000, value=1000, step=100)
        with col_mc2:
            discount_rate_std = st.number_input("Discount Rate Std Dev", value=0.005, step=0.001)
        with col_mc3:
            fcf_growth_std = st.number_input("FCF Growth Std Dev", value=0.005, step=0.001)
            terminal_growth_std = st.number_input("Terminal Growth Std Dev", value=0.002, step=0.001)

# ----- Tabs for Modules -----
tabs = st.tabs([
    "DCF", "Multiples", "DDM", "Balance Sheet Analysis", "EPV", 
    "NAV", "Peer Comparison", "EVA Calculator", "ROIC vs WACC", 
    "Historical Trends", "Consensus Report"
])

# ----- DCF Tab -----
with tabs[0]:
    st.markdown("### Discounted Cash Flow (DCF) Analysis")
    latest_fcf = fetch_latest_fcf(ticker)
    if latest_fcf is None:
        st.error("Unable to retrieve FCF data. Please check the ticker.")
    else:
        hist_fcf_df = get_last_five_fcf(ticker)
        if hist_fcf_df is not None:
            st.markdown("**Historical Free Cash Flow (Last 5 Reporting Periods):**")
            st.table(hist_fcf_df)
        else:
            st.warning("Historical FCF data not available.")
        try:
            ev, projected_fcfs, pv_fcfs, terminal_value, pv_terminal = simple_dcf(
                latest_fcf,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
                fcf_growth=manual_fcf_growth,
                terminal_method=terminal_method,
                terminal_growth=terminal_growth if terminal_growth is not None else 0.0,
                exit_multiple=exit_multiple if exit_multiple is not None else 0.0
            )
            st.markdown("#### Estimated Enterprise Value")
            display_metric_card("DCF EV", ev)
            metrics = fetch_additional_metrics(ticker)
            market_cap = metrics.get("Market Cap")
            if market_cap != "N/A" and isinstance(market_cap, (int, float)):
                st.markdown("#### Enterprise Value Comparison")
                display_ev_comparison(ev, market_cap)
            else:
                st.info("Market Cap data unavailable for comparison.")
            st.markdown("#### Yearly Projections")
            years = list(range(1, forecast_years + 1))
            df_proj = pd.DataFrame({
                "Year": years,
                "Projected FCF": projected_fcfs,
                "Discount Factor": [1 / ((1 + discount_rate) ** year) for year in years],
                "Present Value": pv_fcfs
            })
            st.dataframe(df_proj)
            csv = to_csv(df_proj)
            st.download_button("Download Projections as CSV", data=csv, file_name='projections.csv', mime='text/csv')
            st.markdown("#### FCF Projections Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(years, projected_fcfs, marker='o', color="#2980b9", label='Projected FCF')
            ax.plot(years, pv_fcfs, marker='o', color="#27ae60", label='Discounted FCF')
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_title("FCF vs. Discounted FCF")
            ax.legend()
            st.pyplot(fig)
            if sensitivity_enabled:
                st.markdown("#### Sensitivity Analysis")
                base_terminal_growth = terminal_growth if terminal_growth is not None else 0.02
                sensitivity_df = calculate_sensitivity(
                    latest_fcf=latest_fcf,
                    forecast_years=forecast_years,
                    fcf_growth=manual_fcf_growth,
                    terminal_method=terminal_method,
                    exit_multiple=exit_multiple if exit_multiple is not None else 0.0,
                    base_discount_rate=discount_rate,
                    base_terminal_growth=base_terminal_growth,
                    discount_range=discount_range,
                    growth_range=growth_range
                )
                st.dataframe(sensitivity_df)
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                sns.heatmap(sensitivity_df, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax3, annot_kws={"size":8})
                ax3.set_xlabel("Terminal Growth Rate")
                ax3.set_ylabel("Discount Rate")
                ax3.set_title("Sensitivity of EV")
                st.pyplot(fig3)
            if monte_carlo_enabled:
                st.markdown("#### Monte Carlo Simulation")
                ev_distribution = monte_carlo_dcf(latest_fcf, discount_rate, forecast_years, manual_fcf_growth,
                                                  terminal_method, terminal_growth if terminal_growth is not None else 0.0,
                                                  exit_multiple=exit_multiple if exit_multiple is not None else 0.0,
                                                  iterations=int(iterations),
                                                  discount_rate_std=discount_rate_std,
                                                  fcf_growth_std=fcf_growth_std,
                                                  terminal_growth_std=terminal_growth_std)
                if ev_distribution:
                    st.write(f"Simulated {len(ev_distribution)} iterations")
                    df_sim = pd.DataFrame(ev_distribution, columns=["Simulated EV"])
                    st.bar_chart(df_sim)
                    fig_mc = px.histogram(df_sim, x="Simulated EV", nbins=30,
                                          title="Monte Carlo EV Distribution",
                                          template="plotly_white")
                    st.plotly_chart(fig_mc)
            dcf_value = ev
        except Exception as e:
            st.error(f"DCF Analysis error: {e}")

# ----- Multiples Analysis Tab -----
with tabs[1]:
    st.markdown("### Multiples Analysis")
    multiples = multiples_analysis(ticker)
    if multiples:
        st.table(pd.DataFrame(multiples, index=[0]).T.rename(columns={0:"Value"}))
    else:
        st.warning("Multiples data not available.")

# ----- DDM Tab -----
with tabs[2]:
    st.markdown("### Dividend Discount Model (DDM) Analysis")
    stock = yf.Ticker(ticker)
    info = stock.info
    market_price = info.get("regularMarketPrice", info.get("previousClose", None))
    if market_price:
        st.markdown(f"**Current Market Price:** ${market_price:,.2f}")
    dividend_rate = info.get("dividendRate", None)
    if dividend_rate is None or dividend_rate == 0:
        st.warning("No dividend data available for DDM analysis.")
        ddm_value = None
    else:
        required_return = discount_rate
        ddm_growth = st.number_input("Dividend Growth Rate", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
        ddm_value = ddm_valuation(ticker, required_return, ddm_growth)
        if ddm_value:
            st.markdown("#### DDM Intrinsic Value")
            display_metric_card("DDM Value", ddm_value)
            if market_price:
                diff = ddm_value - market_price
                st.markdown("#### DDM Comparison with Market Price")
                display_metric_card("Difference", diff, unit="", value_color="#27ae60" if diff>=0 else "#e74c3c")
        else:
            st.warning("DDM calculation failed.")
    st.table(pd.DataFrame({"Dividend Rate": [dividend_rate], "Dividend Yield": [info.get('dividendYield', None)]}))

# ----- Balance Sheet Analysis Tab -----
with tabs[3]:
    st.markdown("### Balance Sheet Analysis")
    bs_data = balance_sheet_analysis(ticker)
    if bs_data:
        df_bs = pd.DataFrame.from_dict(bs_data, orient='index', columns=["Value"])
        st.table(df_bs)
        st.markdown("#### Key Metrics")
        key_metrics = {
            "Total Assets": bs_data.get("Total Assets", None),
            "Total Liabilities": bs_data.get("Total Liabilities", None),
            "Stockholders Equity": bs_data.get("Stockholders Equity", None),
            "Working Capital": bs_data.get("Working Capital", None),
            "Current Ratio": bs_data.get("Current Ratio", None),
            "Debt-to-Equity Ratio": bs_data.get("Debt-to-Equity Ratio", None),
            "Debt-to-Asset Ratio": bs_data.get("Debt-to-Asset Ratio", None),
            "Equity Ratio": bs_data.get("Equity Ratio", None),
            "Book Value Per Share": bs_data.get("Book Value Per Share", None),
            "Tangible Book Value Per Share": bs_data.get("Tangible Book Value Per Share", None)
        }
        key_metrics = {k: v for k, v in key_metrics.items() if v is not None}
        display_metric_cards(key_metrics, cols=3)
        if bs_data.get("Total Assets") and bs_data.get("Total Liabilities") and bs_data.get("Stockholders Equity"):
            df_bar = pd.DataFrame({
                "Category": ["Total Assets", "Total Liabilities", "Stockholders Equity"],
                "Value": [bs_data["Total Assets"], bs_data["Total Liabilities"], bs_data["Stockholders Equity"]]
            })
            fig_bar = px.bar(df_bar, x="Category", y="Value", title="Key Balance Sheet Components",
                             color="Category", template="plotly_white")
            st.plotly_chart(fig_bar)
    else:
        st.warning("Balance sheet data not available or incomplete.")

# ----- EPV Tab -----
with tabs[4]:
    st.markdown("### Earnings Power Value (EPV) Analysis")
    epv_val = epv_valuation(ticker, discount_rate)
    if epv_val:
        st.markdown(f"**EPV:** ${epv_val:,.2f}")
        display_metric_card("EPV", epv_val)
    else:
        st.warning("EPV calculation not available. Ensure financial data is complete.")

# ----- NAV Tab -----
with tabs[5]:
    st.markdown("### Net Asset Value (NAV) Analysis")
    nav_val = nav_valuation(ticker)
    if nav_val:
        st.markdown(f"**NAV:** ${nav_val:,.2f}")
        display_metric_card("NAV", nav_val)
    else:
        st.warning("NAV calculation not available. Ensure balance sheet data is complete.")

# ----- Peer Comparison Tab -----
with tabs[6]:
    st.markdown("### Peer Comparison Dashboard")
    peers_input = st.text_input("Enter Peer Tickers (comma separated)", value="MSFT,GOOGL,AMZN")
    peer_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]
    if peer_list:
        data = []
        for p in peer_list:
            m = multiples_analysis(p)
            if m:
                row = {"Ticker": p}
                row.update(m)
                data.append(row)
        if data:
            df_peers = pd.DataFrame(data)
            st.table(df_peers)
            if "P/E" in df_peers.columns and "EV/EBITDA" in df_peers.columns:
                fig_peer = px.scatter(df_peers, x="P/E", y="EV/EBITDA", text="Ticker", title="Peer Comparison: P/E vs EV/EBITDA", template="plotly_white")
                st.plotly_chart(fig_peer)
        else:
            st.warning("No data available for peers.")
    else:
        st.warning("Please enter at least one peer ticker.")

# ----- EVA Calculator Tab -----
with tabs[7]:
    st.markdown("### Economic Value Added (EVA) Calculator")
    eva_result = eva_calculator(ticker, discount_rate, tax_rate)
    if eva_result:
        eva, nopat, invested_capital = eva_result
        st.markdown(f"**NOPAT:** ${nopat:,.2f}")
        st.markdown(f"**Invested Capital:** ${invested_capital:,.2f}")
        st.markdown(f"**EVA:** ${eva:,.2f}")
        display_metric_card("EVA", eva)
    else:
        st.warning("EVA calculation not available. Ensure financial and balance sheet data are complete.")

# ----- ROIC vs WACC Tab -----
with tabs[8]:
    st.markdown("### ROIC vs. WACC Analysis")
    roic_result = roic_analysis(ticker, discount_rate, tax_rate)
    if roic_result:
        roic, wacc = roic_result
        st.markdown(f"**ROIC:** {roic*100:.2f}%")
        st.markdown(f"**WACC:** {wacc*100:.2f}%")
        df_roic = pd.DataFrame({"Metric": ["ROIC", "WACC"], "Value": [roic*100, wacc*100]})
        fig_roic = px.bar(df_roic, x="Metric", y="Value", title="ROIC vs. WACC (%)", color="Metric", template="plotly_white")
        st.plotly_chart(fig_roic)
        display_metric_card("ROIC", roic*100, unit="", value_color="#27ae60")
    else:
        st.warning("ROIC calculation not available. Ensure necessary data are complete.")

# ----- Historical Trends Tab -----
with tabs[9]:
    st.markdown("### Historical Trends Analysis")
    trends_df = historical_trends_analysis(ticker)
    if trends_df is not None and not trends_df.empty:
        st.line_chart(trends_df)
        st.table(trends_df)
    else:
        st.warning("Historical trends data not available.")

# ----- Consensus Report Tab -----
with tabs[10]:
    st.markdown("### Consensus Valuation Report")
    report_data = {}
    report_data["ticker"] = ticker
    report_data["DCF"] = dcf_value if 'dcf_value' in locals() else None
    report_data["DDM"] = ddm_value if 'ddm_value' in locals() else None
    report_data["EPV"] = epv_val if epv_val else None
    report_data["NAV"] = nav_val if nav_val else None
    bs_data = balance_sheet_analysis(ticker)
    report_data["BalanceSheet"] = bs_data if bs_data else None
    # For peer comparison, take average P/E if available:
    if peer_list:
        peer_data = []
        for p in peer_list:
            m = multiples_analysis(p)
            if m and m.get("P/E"):
                peer_data.append(m["P/E"])
        if peer_data:
            report_data["Peer Avg P/E"] = np.mean(peer_data)
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get("marketCap", None)
    report_data["Market Cap"] = market_cap
    if market_cap and dcf_value:
        report_data["Difference"] = dcf_value - market_cap
    if st.button("Generate PDF Report"):
        # For demonstration, we create a dummy chart_images dict (you can attach real charts)
        chart_images = {}
        pdf_bytes = generate_pdf_report(report_data, chart_images)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="Consensus_Report.pdf", mime="application/pdf")
    st.markdown("#### Summary of Valuation Metrics")
    consensus_df = pd.DataFrame({
        "Method": ["DCF", "DDM", "EPV", "NAV", "Market Cap", "Peer Avg P/E"],
        "Value": [report_data["DCF"], report_data["DDM"], report_data["EPV"], report_data["NAV"], report_data["Market Cap"], report_data.get("Peer Avg P/E", "N/A")]
    })
    st.table(consensus_df)
    if market_cap and dcf_value:
        st.markdown(f"**Difference (DCF EV - Market Cap):** ${report_data['Difference']:,.2f}")
