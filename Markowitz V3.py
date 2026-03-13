# Este código retorna a carteira de ações com maior retorno, menor volatilidade e maior Sharpe Ratio
# dado um conjunto de ativos, com restrições de peso mínimo e máximo para cada ativo.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parâmetros ---
MINHA_CARTEIRA = [
    "CVCB3.SA", "SAPR3.SA", "PETR4.SA", "BBAS3.SA",
    "TAEE11.SA", "CPLE3.SA", "CSMG3.SA", "VALE3.SA", "BRSR6.SA"
]
NUM_PORTFOLIOS = 1_000_000
RISK_FREE_RATE = 0.15
MIN_WEIGHT = 0
MAX_WEIGHT = 1
USE_EWMA    = True   # True = covariância EWMA (RiskMetrics); False = covariância histórica simples
EWMA_LAMBDA = 0.94   # Padrão RiskMetrics — meia-vida ~11 dias

# --- Download dos dados ---
pf_data = yf.download(MINHA_CARTEIRA, period="10y")['Close']

# Verificação pós-download
missing = [t for t in MINHA_CARTEIRA if t not in pf_data.columns]
if missing:
    print(f"Aviso: tickers não encontrados e removidos: {missing}")
    MINHA_CARTEIRA = [t for t in MINHA_CARTEIRA if t not in missing]
    pf_data = pf_data[MINHA_CARTEIRA]

pf_data = pf_data.dropna()

# --- Cálculo de retorno e risco ---
retorno_log = np.log(pf_data / pf_data.shift(1)).dropna()
retorno_anualizado = retorno_log.mean() * 252

def calc_ewma_cov(ret: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Covariância EWMA (RiskMetrics): Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}·r_{t-1}ᵀ"""
    cov = ret.iloc[:30].cov().values          # inicializa com os primeiros 30 dias
    for i in range(30, len(ret)):
        r = ret.iloc[i].values
        cov = lam * cov + (1 - lam) * np.outer(r, r)
    return pd.DataFrame(cov * 252, index=ret.columns, columns=ret.columns)

if USE_EWMA:
    cov_carteira = calc_ewma_cov(retorno_log, EWMA_LAMBDA)
    print(f"Usando covariância EWMA (λ={EWMA_LAMBDA})")
else:
    cov_carteira = retorno_log.cov() * 252
    print("Usando covariância histórica simples")

num_ativos = len(MINHA_CARTEIRA)
min_weights = np.full(num_ativos, MIN_WEIGHT)
max_weights = np.full(num_ativos, MAX_WEIGHT)

# --- Geração vetorizada de pesos ---
def generate_limited_weights(num_portfolios, num_ativos, min_w, max_w):
    remaining = 1.0 - min_w.sum()
    if remaining < 0:
        raise ValueError("Soma dos pesos mínimos é maior que 1. Ajuste as restrições.")
    max_additional = max_w - min_w

    additional = np.random.dirichlet(np.ones(num_ativos), size=num_portfolios) * remaining

    # Clipping iterativo para respeitar os limites (converge rapidamente)
    for _ in range(10):
        additional = np.clip(additional, 0, max_additional)
        row_sums = additional.sum(axis=1, keepdims=True)
        additional = additional * (remaining / row_sums)

    additional = np.clip(additional, 0, max_additional)
    return min_w + additional

# --- Simulação ---
weights = generate_limited_weights(NUM_PORTFOLIOS, num_ativos, min_weights, max_weights)
returns = weights @ retorno_anualizado.values
volatility = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_carteira.values, weights))
sharpe = (returns - RISK_FREE_RATE) / volatility

# CVaR paramétrico (diário, 95%) — usado para encontrar o portfólio de menor CVaR
z = norm.ppf(0.05)
daily_mean = returns / 252
daily_std = volatility / np.sqrt(252)
cvar_param = -(daily_mean - daily_std * norm.pdf(z) / 0.05)

# Sortino: penaliza apenas volatilidade negativa (downside deviation)
# Processado em batches para evitar matriz (T × 1M) em memória
SORTINO_BATCH = 10_000
ret_vals = retorno_log.values
downside_std = np.empty(NUM_PORTFOLIOS)
total_batches = (NUM_PORTFOLIOS + SORTINO_BATCH - 1) // SORTINO_BATCH
for i, start in enumerate(range(0, NUM_PORTFOLIOS, SORTINO_BATCH)):
    end = min(start + SORTINO_BATCH, NUM_PORTFOLIOS)
    port_ret = ret_vals @ weights[start:end].T        # (T, batch)
    neg = np.minimum(port_ret, 0)
    downside_std[start:end] = np.sqrt(np.mean(neg**2, axis=0)) * np.sqrt(252)
    print(f"\r  Calculando Sortino... {i + 1}/{total_batches} ({(i + 1) / total_batches * 100:.0f}%)", end="", flush=True)
print()
sortino = (returns - RISK_FREE_RATE) / downside_std

# --- DataFrame com pesos em colunas individuais ---
portfolios = pd.DataFrame({'Returns': returns, 'Volatility': volatility, 'Sharpe': sharpe, 'Sortino': sortino, 'CVaR_param': cvar_param})
for i, ticker in enumerate(MINHA_CARTEIRA):
    portfolios[ticker] = weights[:, i]

# --- Portfólios ótimos ---
max_return_idx  = portfolios['Returns'].idxmax()
min_vol_idx     = portfolios['Volatility'].idxmin()
max_sharpe_idx  = portfolios['Sharpe'].idxmax()
max_sortino_idx = portfolios['Sortino'].idxmax()
min_cvar_idx    = portfolios['CVaR_param'].idxmin()

# --- VaR, CVaR e Drawdown máximo históricos para os portfólios ótimos ---
def calc_var_cvar_hist(w: np.ndarray, retorno_log: pd.DataFrame, confidence: float = 0.95):
    port_returns = retorno_log.values @ w
    var  = -np.percentile(port_returns, (1 - confidence) * 100)
    cvar = -port_returns[port_returns <= -var].mean()
    return var, cvar

def calc_max_drawdown(w: np.ndarray, retorno_log: pd.DataFrame) -> float:
    port_returns = retorno_log.values @ w
    cumulative = np.cumprod(1 + port_returns)          # valor acumulado (base 1)
    running_max = np.maximum.accumulate(cumulative)    # pico até cada dia
    drawdowns = (cumulative - running_max) / running_max
    return drawdowns.min()                             # pior queda (número negativo)

# --- Plot com destaques ---
fig, ax = plt.subplots(figsize=(10, 6))
portfolios.plot(
    x='Volatility', y='Returns', kind='scatter', ax=ax,
    c='Sharpe', cmap='viridis', alpha=0.3, s=1, grid=True
)

otimos = {
    'Maior Retorno':       (max_return_idx,  'red',    '^'),
    'Menor Volatilidade':  (min_vol_idx,     'blue',   'o'),
    'Maior Sharpe Ratio':  (max_sharpe_idx,  'green',  '*'),
    'Maior Sortino':       (max_sortino_idx, 'purple', 'P'),
    'Menor CVaR':          (min_cvar_idx,    'orange', 'D'),
}
for label, (idx, color, marker) in otimos.items():
    p = portfolios.loc[idx]
    ax.scatter(p['Volatility'], p['Returns'], c=color, marker=marker, s=200, zorder=5, label=label)

ax.set_title('Fronteira Eficiente — Markowitz')
ax.set_xlabel('Volatilidade')
ax.set_ylabel('Retorno')
ax.legend()
plt.tight_layout()
plt.show()

# --- Resultados ---
def print_portfolio_info(title, portfolio, assets, w: np.ndarray, retorno_log: pd.DataFrame):
    var_hist, cvar_hist = calc_var_cvar_hist(w, retorno_log)
    max_dd = calc_max_drawdown(w, retorno_log)
    print(f"\n{title}:")
    print(portfolio[['Returns', 'Volatility', 'Sharpe', 'Sortino']])
    print(f"  VaR histórico  (95%, diário): {var_hist:.4f} ({var_hist*100:.2f}%)")
    print(f"  CVaR histórico (95%, diário): {cvar_hist:.4f} ({cvar_hist*100:.2f}%)")
    print(f"  Drawdown máximo (histórico):  {max_dd:.4f} ({max_dd*100:.2f}%)")
    print("Composição:")
    for asset in assets:
        print(f"  {asset}: {portfolio[asset]:.4f}")

for title, idx in [
    ("Portfolio com maior retorno",       max_return_idx),
    ("Portfolio com menor volatilidade",  min_vol_idx),
    ("Portfolio com maior Sharpe Ratio",  max_sharpe_idx),
    ("Portfolio com maior Sortino",       max_sortino_idx),
    ("Portfolio com menor CVaR",          min_cvar_idx),
]:
    w = portfolios.loc[idx, MINHA_CARTEIRA].values
    print_portfolio_info(title, portfolios.loc[idx], MINHA_CARTEIRA, w, retorno_log)
