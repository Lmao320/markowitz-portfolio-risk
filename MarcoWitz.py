import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ===============================
# 1️⃣ Descargar datos históricos
# ===============================
tickers = ['^GSPC', '^NDX', 'GC=F', 'BTC-USD']
names = ['SP500', 'NASDAQ100', 'GOLD', 'BTC']

data = yf.download(tickers, start='2020-06-01', end='2025-06-17')['Close']
data.columns = names
data.dropna(inplace=True)

print("Datos descargados:\n", data.head())

# ===============================
# 2️⃣ Retornos diarios
# ===============================
returns = data.pct_change().dropna()

# ===============================
# 3️⃣ Matriz de correlación
# ===============================
plt.figure(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación de Retornos Diarios")
plt.show()

# ===============================
# 4️⃣ Simulación de portafolios
# ===============================
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

annual_returns = returns.mean() * 252
cov_matrix_annual = returns.cov() * 252

for i in range(num_portfolios):

    weights = np.random.random(len(names))
    weights /= np.sum(weights)
    weights_record.append(weights)

    port_return = np.dot(weights, annual_returns)

    port_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    )

    sharpe = port_return / port_volatility

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe

# ===============================
# 5️⃣ Frontera eficiente
# ===============================
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    results[1, :],
    results[0, :],
    c=results[2, :],
    cmap='viridis',
    s=10,
    alpha=0.4
)

plt.xlabel('Volatilidad Anual')
plt.ylabel('Rentabilidad Anual')
plt.title('Frontera Eficiente - Portafolios Simulados')
plt.colorbar(scatter, label='Ratio Sharpe')
plt.grid(True)
plt.show()

# ===============================
# 6️⃣ Portafolio óptimo (Máx Sharpe)
# ===============================
max_sharpe_idx = results[2].argmax()
opt_weights = weights_record[max_sharpe_idx]

print("\n--- Pesos del Portafolio Óptimo (Sharpe Máximo) ---")
opt_weights_series = pd.Series(opt_weights, index=returns.columns)
print(opt_weights_series.apply(lambda x: f"{x*100:.2f}%"))

opt_port_returns = returns.dot(opt_weights)

annual_return = np.mean(opt_port_returns) * 252
annual_volatility = np.std(opt_port_returns) * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility

print("\n--- Métricas del Portafolio Óptimo ---")
print(f"Rentabilidad anual: {annual_return:.2%}")
print(f"Volatilidad anual: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# ===============================
# 7️⃣ Alpha y Beta
# ===============================
market_returns = returns['SP500']

X = sm.add_constant(market_returns)
model = sm.OLS(opt_port_returns, X).fit()

alpha_daily = model.params['const']
beta = model.params['SP500']
alpha_annual = alpha_daily * 252

print("\n--- Alpha y Beta ---")
print(f"Alpha Diario: {alpha_daily:.6f}")
print(f"Alpha Anualizado: {alpha_annual:.4f} ({alpha_annual*100:.2f}%)")
print(f"Beta frente al S&P 500: {beta:.4f}")

# ===============================
# 8️⃣ Value at Risk
# ===============================
VaR_95 = np.percentile(opt_port_returns, 5)
VaR_99 = np.percentile(opt_port_returns, 1)

print("\n--- Value at Risk (VaR) ---")
print(f"VaR 95% Diario: {VaR_95:.2%}")
print(f"VaR 99% Diario: {VaR_99:.2%}")

# ===============================
# 9️⃣ Último año (Drawdown)
# ===============================
last_date = opt_port_returns.index[-1]
one_year_ago = last_date - pd.DateOffset(years=1)

last_year_returns = opt_port_returns[opt_port_returns.index >= one_year_ago]

cumulative_returns = (1 + last_year_returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max

max_drawdown = drawdown.min()
max_profit = cumulative_returns.max() - 1

print("\n--- Último año (Profit & Drawdown) ---")
print(f"Máximo Profit en el último año: {max_profit:.2%}")
print(f"Máximo Drawdown en el último año: {max_drawdown:.2%}")

# ===============================
# 🔟 Simulación Monte Carlo
# ===============================
num_days = 252
num_sims = 50
last_prices = data.iloc[-1]

mu = returns.mean()
sigma = returns.std()

plt.figure(figsize=(14, 8))

for asset in data.columns:

    S0 = last_prices[asset]
    drift = mu[asset] - 0.5 * sigma[asset]**2

    for i in range(num_sims):

        daily_shocks = np.random.normal(0, sigma[asset], num_days)
        path = S0 * np.exp(np.cumsum(drift + daily_shocks))

        plt.plot(path, alpha=0.3, linewidth=1)

plt.title("Simulación Monte Carlo - Trayectorias de Precios")
plt.xlabel("Días")
plt.ylabel("Precio Simulado")
plt.grid(True)
plt.show()