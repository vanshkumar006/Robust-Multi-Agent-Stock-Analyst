"""
Robust Multi-Agent Stock Analyst System - NO API KEY REQUIRED
==============================================================
This version works completely without any API keys!
Uses rule-based AI logic instead of Gemini for synthesis.
"""

import os
import sys
import subprocess
import warnings
from typing import Dict, List, Any, Optional
from datetime import timedelta, datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SYSTEM INITIALIZATION (NO API NEEDED)
# ============================================================================

class SystemInitializer:
    """Handles library installation - NO API CONFIGURATION NEEDED."""
    
    @staticmethod
    def install_libraries():
        """Auto-install required packages if missing."""
        packages = [
            "yfinance",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "numpy"
        ]
        
        print("🔧 Checking dependencies...")
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                print(f"   Installing {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "-q", "install", "-U", package]
                )
        print("✅ All dependencies ready\n")


# ============================================================================
# 2. TICKER RESOLUTION AGENT
# ============================================================================

class TickerResolver:
    """Resolves company names to ticker symbols and validates them."""
    
    @staticmethod
    def resolve_ticker(query: str) -> str:
        """
        Attempts to find a valid ticker symbol from company name or validates ticker.
        """
        try:
            # First, try direct ticker validation
            test_stock = yf.Ticker(query.upper())
            test_data = test_stock.history(period="1d")
            if not test_data.empty:
                return query.upper()
            
            # If direct ticker fails, try search
            search = yf.Search(query, max_results=1)
            if search.quotes:
                return search.quotes[0]['symbol']
        except Exception:
            pass
        
        return query.upper()


# ============================================================================
# 3. MACHINE LEARNING FORECAST AGENT
# ============================================================================

class MLForecastAgent:
    """Advanced ML agent for stock price forecasting."""
    
    def __init__(self):
        self.model = None
        self.feature_cols = ['Close', 'EMA_12', 'EMA_26', 'RSI', 'Volume', 'Volatility']
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['Target_Next_Day'] = df['Close'].shift(-1)
        
        return df
    
    def get_detailed_ml_forecast(self, ticker: str) -> Dict[str, Any]:
        """Calculates day-by-day 14-day forecast using recursive prediction."""
        try:
            resolved_symbol = TickerResolver.resolve_ticker(ticker)
            stock = yf.Ticker(resolved_symbol)
            df = stock.history(period="2y")
            
            if df.empty or len(df) < 100:
                return {
                    "error": "Insufficient historical data",
                    "ticker": resolved_symbol
                }
            
            df = self._calculate_technical_indicators(df)
            clean_df = df.dropna()
            X = clean_df[self.feature_cols]
            y = clean_df['Target_Next_Day']
            
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X[:-1], y[:-1])
            
            # Recursive 14-day prediction
            predictions_14d = []
            current_features = X.iloc[[-1]].values.copy()
            last_close = df['Close'].iloc[-1]
            
            for day in range(14):
                pred = self.model.predict(current_features)[0]
                predictions_14d.append(round(float(pred), 2))
                new_row = current_features[0].copy()
                new_row[0] = pred
                current_features = [new_row]
            
            trend_factor = predictions_14d[-1] / predictions_14d[0] if predictions_14d[0] != 0 else 1
            forecast_30d = predictions_14d[-1] * (trend_factor ** 1.1)
            historical_accuracy = self._calculate_historical_accuracy(df, X, y)
            
            return {
                "ticker": resolved_symbol,
                "current_price": round(last_close, 2),
                "forecast_14d": {f"Day_{i+1}": val for i, val in enumerate(predictions_14d)},
                "forecast_30d_price": round(forecast_30d, 2),
                "expected_1mo_return": f"{round(((forecast_30d - last_close) / last_close) * 100, 2)}%",
                "expected_2wk_return": f"{round(((predictions_14d[-1] - last_close) / last_close) * 100, 2)}%",
                "confidence_score": historical_accuracy,
                "current_rsi": round(df['RSI'].iloc[-1], 2),
                "trend_strength": self._assess_trend(df)
            }
            
        except Exception as e:
            return {"error": f"ML Analysis failed: {str(e)}"}
    
    def _calculate_historical_accuracy(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate model's historical prediction accuracy."""
        try:
            if len(X) < 60:
                return "Medium"
            
            X_train, y_train = X[:-30], y[:-30]
            X_test, y_test = X[-30:], y[-30:]
            
            test_model = RandomForestRegressor(n_estimators=100, random_state=42)
            test_model.fit(X_train, y_train)
            
            predictions = test_model.predict(X_test)
            actual = y_test.values
            
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            if mape < 5:
                return "High (MAPE < 5%)"
            elif mape < 10:
                return "Medium (MAPE 5-10%)"
            else:
                return "Low (MAPE > 10%)"
        except:
            return "Medium (Based on 2yr History)"
    
    def _assess_trend(self, df: pd.DataFrame) -> str:
        """Assess current market trend strength."""
        try:
            close = df['Close'].iloc[-1]
            sma_10 = df['SMA_10'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            
            if close > sma_10 > sma_50 > sma_200:
                return "Strong Bullish"
            elif close > sma_50 > sma_200:
                return "Bullish"
            elif close < sma_10 < sma_50 < sma_200:
                return "Strong Bearish"
            elif close < sma_50 < sma_200:
                return "Bearish"
            else:
                return "Neutral/Sideways"
        except:
            return "Unknown"


# ============================================================================
# 4. RISK METRICS AGENT
# ============================================================================

class RiskMetricsAgent:
    """Calculates institutional-grade risk metrics."""
    
    @staticmethod
    def get_risk_metrics(ticker: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            resolved_symbol = TickerResolver.resolve_ticker(ticker)
            stock = yf.Ticker(resolved_symbol)
            df = stock.history(period="2y")
            
            if df.empty or len(df) < 252:
                return {"error": "Insufficient data for risk calculation"}
            
            returns = df['Close'].pct_change().dropna()
            
            # Risk metrics
            risk_free_rate = 0.04
            excess_returns = returns.mean() - (risk_free_rate / 252)
            sharpe_ratio = (excess_returns / returns.std()) * np.sqrt(252)
            
            rolling_max = df['Close'].cummax()
            drawdown = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            annual_volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5)
            
            # Beta calculation
            try:
                spy = yf.Ticker("SPY").history(period="2y")
                spy_returns = spy['Close'].pct_change().dropna()
                common_dates = returns.index.intersection(spy_returns.index)
                
                if len(common_dates) > 252:
                    stock_aligned = returns.loc[common_dates]
                    spy_aligned = spy_returns.loc[common_dates]
                    covariance = np.cov(stock_aligned, spy_aligned)[0][1]
                    spy_variance = np.var(spy_aligned)
                    beta = covariance / spy_variance if spy_variance != 0 else 1.0
                else:
                    beta = None
            except:
                beta = None
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            sortino_ratio = (excess_returns / downside_std) * np.sqrt(252)
            
            price_30d_ago = df['Close'].iloc[-30] if len(df) >= 30 else df['Close'].iloc[0]
            sma_50_past = df['Close'].rolling(50).mean().iloc[-30] if len(df) >= 80 else None
            
            if sma_50_past is not None:
                was_bullish = price_30d_ago > sma_50_past
                is_up_now = df['Close'].iloc[-1] > price_30d_ago
                consistency = "High" if (was_bullish == is_up_now) else "Low/Volatile"
            else:
                consistency = "Insufficient Data"
            
            return {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sortino_ratio": round(sortino_ratio, 2),
                "max_drawdown": f"{round(max_drawdown * 100, 2)}%",
                "annual_volatility": f"{round(annual_volatility * 100, 2)}%",
                "var_95": f"{round(var_95 * 100, 2)}%",
                "beta": round(beta, 2) if beta is not None else "N/A",
                "historical_trend_consistency": consistency,
                "risk_category": RiskMetricsAgent._categorize_risk(annual_volatility, sharpe_ratio)
            }
            
        except Exception as e:
            return {"error": f"Risk calculation failed: {str(e)}"}
    
    @staticmethod
    def _categorize_risk(volatility: float, sharpe: float) -> str:
        """Categorize overall risk level."""
        if volatility > 0.4:
            return "High Risk"
        elif volatility > 0.25:
            if sharpe > 1.0:
                return "Medium-High Risk (Good Return)"
            else:
                return "Medium-High Risk"
        elif volatility > 0.15:
            return "Medium Risk"
        else:
            return "Low Risk"


# ============================================================================
# 5. MARKET INTELLIGENCE AGENT
# ============================================================================

class MarketIntelligenceAgent:
    """Gathers market sentiment, news, insider trading, and fundamentals."""
    
    @staticmethod
    def get_market_intelligence(ticker: str) -> Dict[str, Any]:
        """Scrape comprehensive market intelligence."""
        try:
            resolved_symbol = TickerResolver.resolve_ticker(ticker)
            stock = yf.Ticker(resolved_symbol)
            info = stock.info
            
            # News
            try:
                news = stock.news[:5]
                headlines = [n['title'] for n in news] if news else ["No recent news available"]
                news_dates = [datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d') 
                            for n in news] if news else []
            except:
                headlines = ["News data unavailable"]
                news_dates = []
            
            # Insiders
            insider_status = "No recent data"
            insider_details = []
            try:
                insiders = stock.insider_transactions
                if insiders is not None and not insiders.empty:
                    recent = insiders.head(10)
                    buys = len(recent[recent['Transaction'].str.contains('Purchase', case=False, na=False)])
                    sells = len(recent[recent['Transaction'].str.contains('Sale', case=False, na=False)])
                    insider_status = f"{buys} Buys vs {sells} Sells (Last 10 Transactions)"
                    
                    for _, row in recent.head(3).iterrows():
                        insider_details.append({
                            "insider": row.get('Insider', 'Unknown'),
                            "transaction": row.get('Transaction', 'Unknown'),
                            "shares": row.get('Shares', 'Unknown')
                        })
            except:
                pass
            
            # Fundamentals
            fundamentals = {
                "Market_Cap": f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A",
                "P_E_Ratio": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A",
                "Forward_P_E": round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else "N/A",
                "P_S_Ratio": round(info.get('priceToSalesTrailing12Months', 0), 2) if info.get('priceToSalesTrailing12Months') else "N/A",
                "P_B_Ratio": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else "N/A",
                "EV_EBITDA": round(info.get('enterpriseToEbitda', 0), 2) if info.get('enterpriseToEbitda') else "N/A",
                "Debt_to_Equity": round(info.get('debtToEquity', 0), 2) if info.get('debtToEquity') else "N/A",
                "Profit_Margin": f"{round(info.get('profitMargins', 0) * 100, 2)}%" if info.get('profitMargins') else "N/A",
                "ROE": f"{round(info.get('returnOnEquity', 0) * 100, 2)}%" if info.get('returnOnEquity') else "N/A"
            }
            
            analyst_data = {
                "Target_Price": round(info.get('targetMeanPrice', 0), 2) if info.get('targetMeanPrice') else "N/A",
                "Recommendation": info.get('recommendationKey', 'N/A').upper()
            }
            
            profile = {
                "Sector": info.get('sector', 'N/A'),
                "Industry": info.get('industry', 'N/A'),
                "Country": info.get('country', 'N/A'),
                "Employees": f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "N/A"
            }
            
            return {
                "ticker": resolved_symbol,
                "company_name": info.get('longName', resolved_symbol),
                "news_headlines": list(zip(news_dates, headlines)) if news_dates else headlines,
                "insider_sentiment": insider_status,
                "insider_details": insider_details,
                "fundamentals": fundamentals,
                "analyst_data": analyst_data,
                "company_profile": profile,
                "business_summary": info.get('longBusinessSummary', 'N/A')[:300] + "..."
            }
            
        except Exception as e:
            return {"error": f"Market intelligence gathering failed: {str(e)}"}


# ============================================================================
# 6. VISUALIZATION AGENT
# ============================================================================

class VisualizationAgent:
    """Creates professional financial charts."""
    
    @staticmethod
    def generate_precision_chart(ticker: str, forecast_data: Optional[Dict] = None) -> str:
        """Generate comprehensive stock chart."""
        try:
            resolved_symbol = TickerResolver.resolve_ticker(ticker)
            stock = yf.Ticker(resolved_symbol)
            hist = stock.history(period="6mo")
            
            if hist.empty:
                return "No historical data available"
            
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(hist.index, hist['Close'], label='Actual Price',
                    color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.plot(hist.index, hist['SMA_50'], label='50-Day MA',
                    color='#F77F00', linestyle='--', alpha=0.7)
            ax1.plot(hist.index, hist['SMA_200'], label='200-Day MA',
                    color='#D62828', linestyle='--', alpha=0.7)
            
            if forecast_data and 'forecast_14d' in forecast_data and 'error' not in forecast_data:
                future_prices = list(forecast_data['forecast_14d'].values())
                last_date = hist.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                            periods=14, freq='D')
                
                ax1.plot([last_date] + list(future_dates),
                        [hist['Close'].iloc[-1]] + future_prices,
                        label='14-Day ML Forecast', color='#06FFA5',
                        linestyle='--', marker='o', markersize=4, linewidth=2)
                
                upper_band = [p * 1.05 for p in future_prices]
                lower_band = [p * 0.95 for p in future_prices]
                ax1.fill_between(future_dates, lower_band, upper_band,
                            alpha=0.2, color='#06FFA5', label='Confidence Band (±5%)')
            
            ax1.set_title(f'{resolved_symbol} - Precision Technical Analysis',
                        fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=12)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle=':')
            
            colors = ['#26C485' if hist['Close'].iloc[i] >= hist['Close'].iloc[i-1]
                    else '#F6465D' for i in range(1, len(hist))]
            colors.insert(0, '#26C485')
            
            ax2.bar(hist.index, hist['Volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle=':')
            
            plt.tight_layout()
            plt.show()
            
            return f"✅ Chart generated for {resolved_symbol}"
            
        except Exception as e:
            return f"Visualization failed: {str(e)}"


# ============================================================================
# 7. RULE-BASED AI SYNTHESIZER (NO API REQUIRED!)
# ============================================================================

class RuleBasedAISynthesizer:
    """
    Rule-based AI that generates investment reports WITHOUT any API.
    Uses algorithmic logic to create bull/bear debates and recommendations.
    """
    
    @staticmethod
    def generate_investment_report(ticker: str, ml_data: Dict, risk_data: Dict,
                                intel_data: Dict) -> str:
        """Generate comprehensive investment report using rule-based logic."""
        
        # Check for errors
        if 'error' in ml_data or 'error' in risk_data or 'error' in intel_data:
            return f"❌ Unable to generate complete analysis due to data errors:\n" + \
                    f"ML: {ml_data.get('error', 'OK')}\n" + \
                    f"Risk: {risk_data.get('error', 'OK')}\n" + \
                    f"Intel: {intel_data.get('error', 'OK')}"
        
        report = []
        report.append("="*74)
        report.append(f"INVESTMENT ANALYSIS REPORT: {ticker}")
        report.append("="*74)
        report.append("")
        
        # EXECUTIVE SUMMARY
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 74)
        summary = RuleBasedAISynthesizer._generate_executive_summary(ml_data, risk_data, intel_data)
        report.append(summary)
        report.append("")
        
        # QUANTITATIVE ANALYSIS
        report.append("📈 QUANTITATIVE ANALYSIS")
        report.append("-" * 74)
        report.append(f"Current Price: ${ml_data.get('current_price', 'N/A')}")
        report.append(f"14-Day Forecast: {ml_data.get('expected_2wk_return', 'N/A')}")
        report.append(f"30-Day Target: ${ml_data.get('forecast_30d_price', 'N/A')}")
        report.append(f"Expected 1-Month Return: {ml_data.get('expected_1mo_return', 'N/A')}")
        report.append(f"ML Confidence: {ml_data.get('confidence_score', 'N/A')}")
        report.append(f"Current RSI: {ml_data.get('current_rsi', 'N/A')}")
        report.append(f"Trend Strength: {ml_data.get('trend_strength', 'N/A')}")
        report.append("")
        
        # RISK ASSESSMENT
        report.append("⚠️  RISK ASSESSMENT")
        report.append("-" * 74)
        report.append(f"Sharpe Ratio: {risk_data.get('sharpe_ratio', 'N/A')}")
        report.append(f"Sortino Ratio: {risk_data.get('sortino_ratio', 'N/A')}")
        report.append(f"Maximum Drawdown: {risk_data.get('max_drawdown', 'N/A')}")
        report.append(f"Annual Volatility: {risk_data.get('annual_volatility', 'N/A')}")
        report.append(f"Beta: {risk_data.get('beta', 'N/A')}")
        report.append(f"Risk Category: {risk_data.get('risk_category', 'N/A')}")
        report.append("")
        
        # FUNDAMENTAL ANALYSIS
        report.append("💼 FUNDAMENTAL & SENTIMENT ANALYSIS")
        report.append("-" * 74)
        fundamentals = intel_data.get('fundamentals', {})
        report.append(f"Market Cap: {fundamentals.get('Market_Cap', 'N/A')}")
        report.append(f"P/E Ratio: {fundamentals.get('P_E_Ratio', 'N/A')}")
        report.append(f"P/S Ratio: {fundamentals.get('P_S_Ratio', 'N/A')}")
        report.append(f"ROE: {fundamentals.get('ROE', 'N/A')}")
        report.append(f"Insider Activity: {intel_data.get('insider_sentiment', 'N/A')}")
        
        analyst = intel_data.get('analyst_data', {})
        report.append(f"Analyst Target: ${analyst.get('Target_Price', 'N/A')}")
        report.append(f"Analyst Recommendation: {analyst.get('Recommendation', 'N/A')}")
        report.append("")
        
        # BULL CASE
        report.append("🐂 BULL CASE (Long Thesis)")
        report.append("-" * 74)
        bull_points = RuleBasedAISynthesizer._generate_bull_case(ml_data, risk_data, intel_data)
        for point in bull_points:
            report.append(f"  ✓ {point}")
        report.append("")
        
        # BEAR CASE
        report.append("🐻 BEAR CASE (Short Thesis)")
        report.append("-" * 74)
        bear_points = RuleBasedAISynthesizer._generate_bear_case(ml_data, risk_data, intel_data)
        for point in bear_points:
            report.append(f"  ⚠ {point}")
        report.append("")
        
        # FINAL VERDICT
        report.append("⚖️  FINAL VERDICT & RECOMMENDATION")
        report.append("-" * 74)
        verdict = RuleBasedAISynthesizer._generate_verdict(ml_data, risk_data, intel_data,
                                                            bull_points, bear_points)
        report.append(verdict)
        report.append("")
        
        # ACTIONABLE INSIGHTS
        report.append("🎯 ACTIONABLE INSIGHTS")
        report.append("-" * 74)
        insights = RuleBasedAISynthesizer._generate_actionable_insights(ml_data, risk_data, intel_data)
        report.append(insights)
        report.append("")
        
        report.append("="*74)
        report.append("⚠️  DISCLAIMER: This analysis is for educational purposes only.")
        report.append("Always consult a licensed financial advisor before making investment decisions.")
        report.append("="*74)
        
        return "\n".join(report)
    
    @staticmethod
    def _generate_executive_summary(ml_data, risk_data, intel_data) -> str:
        """Generate executive summary based on data."""
        ticker = ml_data.get('ticker', 'UNKNOWN')
        company = intel_data.get('company_name', ticker)
        expected_return = ml_data.get('expected_1mo_return', 'N/A')
        trend = ml_data.get('trend_strength', 'Unknown')
        risk_cat = risk_data.get('risk_category', 'Unknown')
        
        summary = f"{company} ({ticker}) exhibits a {trend.lower()} trend with "
        summary += f"ML models projecting {expected_return} over 30 days. "
        summary += f"Risk profile is {risk_cat.lower()}. "
        
        # Quick assessment
        try:
            ret_val = float(expected_return.replace('%', ''))
            if ret_val > 5:
                summary += "Strong upside potential detected."
            elif ret_val > 0:
                summary += "Moderate upside potential."
            else:
                summary += "Downside risks present."
        except:
            summary += "Returns uncertain."
        
        return summary
    
    @staticmethod
    def _generate_bull_case(ml_data, risk_data, intel_data) -> List[str]:
        """Generate bullish arguments based on data."""
        bull_points = []
        
        # ML forecast bullish?
        try:
            expected_return = float(ml_data.get('expected_1mo_return', '0%').replace('%', ''))
            if expected_return > 0:
                impact = "HIGH" if expected_return > 5 else "MEDIUM"
                bull_points.append(f"ML forecast predicts {expected_return:+.2f}% gain in 30 days - {impact} IMPACT")
        except:
            pass
        
        # Trend strength
        trend = ml_data.get('trend_strength', '')
        if 'Bullish' in trend:
            bull_points.append(f"Strong technical trend ({trend}) supports upward momentum - HIGH IMPACT")
        
        # Sharpe ratio
        try:
            sharpe = risk_data.get('sharpe_ratio', 0)
            if isinstance(sharpe, (int, float)) and sharpe > 1.0:
                impact = "HIGH" if sharpe > 1.5 else "MEDIUM"
                bull_points.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe}) - {impact} IMPACT")
        except:
            pass
        
        # RSI oversold?
        try:
            rsi = ml_data.get('current_rsi', 50)
            if isinstance(rsi, (int, float)) and rsi < 30:
                bull_points.append(f"RSI at {rsi} suggests oversold conditions (reversal opportunity) - MEDIUM IMPACT")
        except:
            pass
        
        # Insider buying
        insider = intel_data.get('insider_sentiment', '')
        if 'Buy' in insider:
            try:
                parts = insider.split()
                buys = int(parts[0])
                if buys > 0:
                    impact = "HIGH" if buys > 3 else "MEDIUM"
                    bull_points.append(f"Insider buying detected ({insider}) - {impact} IMPACT")
            except:
                bull_points.append(f"Insider buying activity observed - MEDIUM IMPACT")
        
        # Analyst recommendation
        analyst_rec = intel_data.get('analyst_data', {}).get('Recommendation', '')
        if analyst_rec in ['STRONG_BUY', 'BUY']:
            bull_points.append(f"Analyst consensus: {analyst_rec} - MEDIUM IMPACT")
        
        # Low volatility
        try:
            vol = risk_data.get('annual_volatility', '100%')
            vol_val = float(vol.replace('%', ''))
            if vol_val < 20:
                bull_points.append(f"Low volatility ({vol}) indicates stable investment - LOW IMPACT")
        except:
            pass
        
        if not bull_points:
            bull_points.append("Limited bullish signals detected in current analysis")
        
        return bull_points
    
    @staticmethod
    def _generate_bear_case(ml_data, risk_data, intel_data) -> List[str]:
        """Generate bearish arguments based on data."""
        bear_points = []
        
        # ML forecast bearish?
        try:
            expected_return = float(ml_data.get('expected_1mo_return', '0%').replace('%', ''))
            if expected_return < 0:
                impact = "HIGH" if expected_return < -5 else "MEDIUM"
                bear_points.append(f"ML forecast predicts {expected_return:.2f}% decline - {impact} IMPACT")
        except:
            pass
        
        # Bearish trend
        trend = ml_data.get('trend_strength', '')
        if 'Bearish' in trend:
            bear_points.append(f"Technical trend is {trend} - HIGH IMPACT")
        
        # High drawdown
        try:
            dd = risk_data.get('max_drawdown', '0%')
            dd_val = float(dd.replace('%', '').replace('-', ''))
            if dd_val > 20:
                impact = "HIGH" if dd_val > 30 else "MEDIUM"
                bear_points.append(f"Significant historical drawdown ({dd}) shows vulnerability - {impact} IMPACT")
        except:
            pass
        
        # High volatility
        try:
            vol = risk_data.get('annual_volatility', '0%')
            vol_val = float(vol.replace('%', ''))
            if vol_val > 30:
                impact = "HIGH" if vol_val > 40 else "MEDIUM"
                bear_points.append(f"High volatility ({vol}) increases risk - {impact} IMPACT")
        except:
            pass
        
        # RSI overbought
        try:
            rsi = ml_data.get('current_rsi', 50)
            if isinstance(rsi, (int, float)) and rsi > 70:
                bear_points.append(f"RSI at {rsi} indicates overbought conditions (correction risk) - MEDIUM IMPACT")
        except:
            pass
        
        # Insider selling
        insider = intel_data.get('insider_sentiment', '')
        if 'Sell' in insider:
            try:
                parts = insider.split()
                if len(parts) >= 5:
                    sells = int(parts[4])
                    if sells > 0:
                        impact = "HIGH" if sells > 3 else "MEDIUM"
                        bear_points.append(f"Insider selling detected ({insider}) - {impact} IMPACT")
            except:
                bear_points.append(f"Insider selling activity observed - MEDIUM IMPACT")
        
        # High valuation
        try:
            pe = intel_data.get('fundamentals', {}).get('P_E_Ratio', 0)
            if isinstance(pe, (int, float)) and pe > 30:
                impact = "MEDIUM" if pe > 50 else "LOW"
                bear_points.append(f"High P/E ratio ({pe}) suggests premium valuation - {impact} IMPACT")
        except:
            pass
        
        # Negative analyst view
        analyst_rec = intel_data.get('analyst_data', {}).get('Recommendation', '')
        if analyst_rec in ['SELL', 'UNDERPERFORM']:
            bear_points.append(f"Analyst consensus: {analyst_rec} - MEDIUM IMPACT")
        
        if not bear_points:
            bear_points.append("Limited bearish signals detected in current analysis")
        
        return bear_points
    
    @staticmethod
    def _generate_verdict(ml_data, risk_data, intel_data, bull_points, bear_points) -> str:
        """Generate final investment verdict."""
        # Score-based system
        bull_score = 0
        bear_score = 0
        
        for point in bull_points:
            if 'HIGH IMPACT' in point:
                bull_score += 3
            elif 'MEDIUM IMPACT' in point:
                bull_score += 2
            else:
                bull_score += 1
        
        for point in bear_points:
            if 'HIGH IMPACT' in point:
                bear_score += 3
            elif 'MEDIUM IMPACT' in point:
                bear_score += 2
            else:
                bear_score += 1
        
        # Determine verdict
        verdict = []
        score_diff = bull_score - bear_score
        
        if score_diff >= 6:
            recommendation = "STRONG BUY"
            confidence = "High"
            position_size = "5-7% of portfolio"
        elif score_diff >= 3:
            recommendation = "BUY"
            confidence = "Medium-High"
            position_size = "3-5% of portfolio"
        elif score_diff >= -2:
            recommendation = "HOLD"
            confidence = "Medium"
            position_size = "Maintain current position"
        elif score_diff >= -5:
            recommendation = "SELL"
            confidence = "Medium-High"
            position_size = "Reduce position"
        else:
            recommendation = "STRONG SELL"
            confidence = "High"
            position_size = "Exit position"
        
        verdict.append(f"Recommendation: **{recommendation}**")
        verdict.append(f"Confidence Level: {confidence}")
        verdict.append(f"Bull Score: {bull_score} | Bear Score: {bear_score}")
        verdict.append(f"Position Sizing: {position_size}")
        verdict.append("")
        verdict.append("Rationale:")
        
        if recommendation in ["STRONG BUY", "BUY"]:
            verdict.append(f"  • Bull case significantly outweighs bear case ({bull_score} vs {bear_score})")
            verdict.append(f"  • {len(bull_points)} bullish factors identified")
            verdict.append("  • Risk-reward ratio favorable for long positions")
        elif recommendation == "HOLD":
            verdict.append("  • Bull and bear cases are balanced")
            verdict.append("  • Wait for clearer signals before action")
            verdict.append("  • Monitor key metrics closely")
        else:
            verdict.append(f"  • Bear case outweighs bull case ({bear_score} vs {bull_score})")
            verdict.append(f"  • {len(bear_points)} bearish factors identified")
            verdict.append("  • Risk-reward ratio unfavorable")
        
        return "\n".join(verdict)
    
    @staticmethod
    def _generate_actionable_insights(ml_data, risk_data, intel_data) -> str:
        """Generate actionable trading insights."""
        insights = []
        
        current_price = ml_data.get('current_price', 0)
        forecast_30d = ml_data.get('forecast_30d_price', current_price)
        
        if isinstance(current_price, (int, float)) and isinstance(forecast_30d, (int, float)):
            # Entry points
            conservative_entry = current_price * 0.98
            aggressive_entry = current_price * 1.02
            insights.append(f"Entry Points:")
            insights.append(f"  • Conservative: ${conservative_entry:.2f} (2% below current)")
            insights.append(f"  • Aggressive: ${aggressive_entry:.2f} (current to +2%)")
            insights.append("")
            
            # Stop loss
            stop_loss = current_price * 0.92
            insights.append(f"Stop Loss: ${stop_loss:.2f} (-8% from current)")
            insights.append("")
            
            # Targets
            conservative_target = current_price + (forecast_30d - current_price) * 0.5
            aggressive_target = forecast_30d
            insights.append(f"Price Targets:")
            insights.append(f"  • Conservative: ${conservative_target:.2f} (+{((conservative_target/current_price-1)*100):.1f}%)")
            insights.append(f"  • Aggressive: ${aggressive_target:.2f} (+{((aggressive_target/current_price-1)*100):.1f}%)")
            insights.append("")
            
            # Time horizon
            insights.append("Time Horizon: 30-60 days for thesis to play out")
            insights.append("")
            
            # Key risks
            insights.append("Key Risks to Monitor:")
            max_dd = risk_data.get('max_drawdown', 'N/A')
            insights.append(f"  • Historical max drawdown: {max_dd}")
            vol = risk_data.get('annual_volatility', 'N/A')
            insights.append(f"  • Volatility level: {vol}")
            insights.append("  • Market sentiment shifts")
            insights.append("  • Sector-wide selloffs")
        else:
            insights.append("Insufficient data for specific entry/exit recommendations")
            insights.append("Focus on risk management and position sizing")
        
        return "\n".join(insights)


# ============================================================================
# 8. MULTI-AGENT ORCHESTRATOR (NO API VERSION)
# ============================================================================

class MultiAgentOrchestrator:
    """Coordinates all agents WITHOUT requiring any API keys."""
    
    def __init__(self):
        self.ml_agent = MLForecastAgent()
        self.risk_agent = RiskMetricsAgent()
        self.intel_agent = MarketIntelligenceAgent()
        self.viz_agent = VisualizationAgent()
        self.synthesizer = RuleBasedAISynthesizer()
        print("✅ Multi-Agent System initialized (NO API REQUIRED)\n")
    
    def analyze_stock(self, ticker: str) -> str:
        """Run complete multi-agent analysis on a stock."""
        try:
            print(f"🔍 Initiating deep analysis for {ticker}...")
            print("📊 Agents: ML Forecast | Risk Metrics | Market Intel | Visualization")
            print("⏳ This may take 15-30 seconds...\n")
            
            # Execute all agents
            ml_data = self.ml_agent.get_detailed_ml_forecast(ticker)
            risk_data = self.risk_agent.get_risk_metrics(ticker)
            intel_data = self.intel_agent.get_market_intelligence(ticker)
            
            # Generate visualization
            self.viz_agent.generate_precision_chart(ticker, ml_data)
            
            # Synthesize report using rule-based AI
            report = self.synthesizer.generate_investment_report(
                ticker, ml_data, risk_data, intel_data
            )
            
            return report
            
        except Exception as e:
            return f"❌ Analysis Error: {str(e)}\n\nPlease try again or check ticker symbol."


# ============================================================================
# 9. MAIN APPLICATION
# ============================================================================

def print_banner():
    """Print system banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🏛️  ROBUST MULTI-AGENT STOCK ANALYST (NO API KEY REQUIRED)  🏛️  ║
║                                                                      ║
║           Institutional-Grade Investment Analysis                   ║
║              100% Free - No API Keys Needed!                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Features:
  ✓ 14-Day Recursive ML Forecasting with Random Forest
  ✓ Comprehensive Risk Metrics (Sharpe, Sortino, Beta, VaR)
  ✓ Real-time Market Sentiment & Insider Trading Analysis
  ✓ Advanced Technical Visualization with Forecast Overlay
  ✓ Rule-Based AI Bull/Bear Debate (NO API REQUIRED!)
  ✓ Institutional-Grade Investment Recommendations

"""
    print(banner)


def main():
    """Main application entry point."""
    print_banner()
    SystemInitializer.install_libraries()
    
    # Initialize orchestrator (NO API NEEDED!)
    orchestrator = MultiAgentOrchestrator()
    
    print("="*74)
    print("System ready. Type 'quit' or 'exit' to terminate.")
    print("💡 NO API KEY REQUIRED - Completely free to use!")
    print("="*74)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n💼 Enter Stock Ticker or Company Name: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Multi-Agent Stock Analyst System!")
                break
            
            if not user_input:
                continue
            
            # Run analysis
            print("\n" + "="*74)
            result = orchestrator.analyze_stock(user_input)
            print(result)
            print("="*74)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again with a different ticker.\n")


if __name__ == "__main__":
    main()
