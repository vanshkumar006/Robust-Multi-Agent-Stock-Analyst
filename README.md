🌟 Project Vision
The system mimics a professional investment committee. Specialized "agents" perform individual tasks—technical analysis, risk management, and fundamental research—before a central Synthesizer generates a final, logic-backed investment report.


🤖 The Multi-Agent Workflow
The system operates using an Orchestrator Pattern, triggering a specialized sequence for every ticker:


🔍 A. Ticker Resolution Agent 
Ensures data integrity by using "fuzzy matching" to resolve company names (e.g., "Apple") into valid exchange symbols (AAPL) and validating them via Yahoo Finance.


📈 B. Machine Learning (ML) Forecast Agent

Model: Random Forest Regressor.

Features: Calculates EMA (12/26), RSI, Volatility, and Moving Averages (10/50/200).

Recursive Forecasting: Performs step-by-step prediction for a 14-day horizon, using each day's output as the next day's input to simulate a realistic price path.

Confidence Scoring: Back-tests against the last 30 days of real data to provide a "High/Medium/Low" reliability rating.


⚖️ C. Risk Metrics Agent

Calculates institutional-grade quantitative ratios:

Sharpe & Sortino Ratios: Measuring reward-to-volatility efficiency.

Maximum Drawdown: Analyzing historical peak-to-trough declines.

Beta: Sensitivity relative to the S&P 500.

Value at Risk (VaR): Potential losses at a 95% confidence interval.


💡 D. Market Intelligence Agent

Fundamentals: Scrapes P/E, Debt-to-Equity, ROE, and Market Cap.

Insider Tracking: Monitors the last 10 transactions by CEOs/Directors to detect "smart money" movement.

Sentiment: Aggregates news headlines and institutional analyst recommendations.


🎨 E. Visualization Agent

Generates a professional dual-pane financial chart:

Upper: Price action, MA ribbons, and the 14-day ML forecast path with a 5% confidence band.

Lower: Color-coded volume bars representing buying/selling pressure.


🧠 The "Brain": Rule-Based AI Synthesizer

The core innovation is a Weighted Scoring Algorithm that simulates human judgment without an LLM:

Scoring Logic: Bullish ML forecasts (+3 pts), High Insider Selling (-3 pts), Overbought RSI (-2 pts), etc.

Conflict Resolution: If the ML is bullish but the trend is "Strong Bearish," the system penalizes confidence and issues a "Hold" verdict.

Actionable Insights: Automatically calculates Conservative/Aggressive Entry Points and Stop-Loss levels based on current volatility.


🚀 Key Innovation: No-API Dependency


By using a local rule-based engine instead of a Generative AI API:

Privacy: No financial queries or portfolio interests are sent to third-party servers.

Zero Cost: 100% free to run indefinitely.

Reliability: No "API Rate Limits" or "Model Downtime."


🛠️ Installation & Requirements

Python 3.8+
Core Libraries:

pip install yfinance scikit-learn pandas numpy matplotlib
Usage

Run the main script and enter a ticker or company name:

python main.py


🎯 Use Cases

Day Traders: Get a data-driven "second opinion" on trade setups.

Long-term Investors: Quickly check the risk health and insider sentiment of core holdings.

Developers: A blueprint for coordinating multiple Python scripts into a cohesive "Agentic" system.


kaggle link : https://www.kaggle.com/code/vanshkumar006/robust-multi-agent-stock-analyst


⚠️ Disclaimer
This project transforms raw market data into a structured investment thesis for educational purposes. It does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions.
