Robust Multi-Agent Stock Analyst

1. Project Vision
The Robust Multi-Agent Stock Analyst is an autonomous financial research environment designed to bridge the gap between retail trading and institutional-grade data analysis. While most modern AI financial tools rely on expensive, third-party Large Language Model (LLM) APIs (like OpenAI or Gemini) to provide insights, this project is built on a fully local, rule-based intelligence framework.
It mimics a professional investment committee where specialized "agents" (scripts) perform individual tasks—technical analysis, risk management, and fundamental research—before a central "Synthesizer" generates a final investment thesis.

2. Technical Architecture & Agent Workflow
The system operates using an Orchestrator Pattern. When a user enters a stock ticker, the following workflow is triggered:

A. Ticker Resolution Agent
Purpose: Ensures data integrity.
Logic: It uses "fuzzy matching" to resolve company names (e.g., "Apple") into valid exchange tickers (AAPL). It validates the ticker against Yahoo Finance to ensure historical data exists before passing it to the next agent.

B. Machine Learning (ML) Forecast Agent
Model: Random Forest Regressor.
Features: It calculates technical indicators including EMA (12/26), RSI (14-day), Volatility (20-day), and Moving Averages (10/50/200).
Recursive Forecasting: Unlike standard models that predict a single point, this agent performs recursive prediction. It predicts "Day 1," then uses that prediction as an input to predict "Day 2," continuing for a 14-day horizon to simulate a price path.
Confidence Scoring: It measures the model's accuracy against the last 30 days of real data to provide a "High/Medium/Low" confidence rating.

C. Risk Metrics Agent
Quantitative Math: It calculates professional risk ratios:
Sharpe Ratio: Reward-to-volatility efficiency.
Sortino Ratio: Focuses specifically on "bad" (downside) volatility.
Maximum Drawdown: The largest peak-to-trough decline in a 2-year period.
Beta: Correlation and sensitivity to the S&P 500 (SPY).
Value at Risk (VaR): Predicts potential losses at a 95% confidence interval.

D. Market Intelligence Agent
Fundamental Analysis: Scrapes P/E ratios, Debt-to-Equity, Return on Equity (ROE), and Market Cap.
Insider Tracking: Analyzes the most recent 10 insider transactions (CEOs/Directors) to determine if the "smart money" is buying or selling.
Sentiment Analysis: Pulls recent news headlines and analyst recommendations (Buy/Hold/Sell) from institutional firms.

E. Visualization Agent
Produces a dual-pane financial chart.
Upper Pane: Price action + Moving Average ribbons + the 14-day ML Forecast path with a 5% confidence shaded band.
Lower Pane: Volume bars color-coded by buying/selling pressure.

3. The "Brain": Rule-Based AI Synthesizer
The most innovative part of this project is the Rule-Based AI Synthesizer. Instead of sending data to an LLM, it uses a Weighted Scoring Algorithm to simulate human judgment:
Fact-Checking: It looks at the outputs of all other agents.
Weighted Scoring:
A Bullish ML Forecast adds +3 points.
High Insider Selling subtracts -3 points.
An RSI over 70 (Overbought) subtracts -2 points.
Conflict Resolution: If the ML Forecast is bullish but the Technical Trend is "Strong Bearish," the synthesizer penalizes the confidence score and issues a "Neutral/Hold" verdict.
Actionable Insights: It automatically calculates Conservative/Aggressive Entry Points and Stop-Loss levels based on the current volatility.

4. Key Innovation: No-API Dependency
By using a rule-based engine instead of a Generative AI API:
Privacy: No financial queries or portfolio interests are sent to third-party servers.
Zero Cost: The system is 100% free to run indefinitely.
Reliability: The system will never fail due to "API Rate Limits" or "Model Downtime."

5. Potential Use Cases
Day Traders: For a quick, data-driven "second opinion" on a setup.
Long-term Investors: For checking the risk health and insider sentiment of a core holding.
Students/Developers: As a blueprint for how to coordinate multiple Python scripts into a cohesive "Agentic" system.

6. Development Requirements
Python 3.8+
Core Libraries: yfinance (Data), scikit-learn (ML), pandas/numpy (Processing), matplotlib (Charts).

Summary of Value
This project transforms raw market data into a structured investment thesis. It doesn't just give you numbers; it gives you a logical argument for why a stock might be a good or bad investment, supported by machine learning and institutional mathematics.
