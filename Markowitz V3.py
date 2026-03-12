# Este código retorna a carteira de ações com maior retorno, menor volatilidade e maior Sharpe Ratio dado um conjunto de ativos, com restrições de peso mínimo e máximo para cada ativo.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Definindo minha carteira de ações
MinhaCarteira = [
    "CVCB3.SA", "SAPR3.SA", "PETR4.SA", "BBAS3.SA",
    "TAEE11.SA", "CPLE6.SA", "CSMG3.SA", "VALE3.SA", "BRSR6.SA"
]

# Baixando os dados de fechamento ajustado
pf_data = yf.download(MinhaCarteira, period="10y")['Close']

# Calculo do Retorno e Risco da Carteira
retorno_log = np.log(pf_data / pf_data.shift(1))
retorno_anualizado = retorno_log.mean() * 252
risco_carteira = retorno_log.std() * np.sqrt(252)
cov_carteira = retorno_log.cov() * 252

num_ativos = len(MinhaCarteira)
num_portfolios = 1000000

# Defina os limites mínimos e máximos para cada ativo (na mesma ordem da carteira)
min_weights = np.array([0.05] * num_ativos)  # Exemplo: mínimo de 5% para cada ativo
max_weights = np.array([0.20] * num_ativos)  # Exemplo: máximo de 20% para cada ativo

# Função para gerar pesos respeitando os limites
def generate_limited_weights(num_portfolios, num_ativos, min_w, max_w):
    weights_list = []
    for _ in range(num_portfolios):
        # Start with minimum weights
        w = min_w.copy()
        # Distribute the remaining weight randomly
        remaining = 1.0 - w.sum()
        # If remaining is negative, constraints are impossible
        if remaining < 0:
            raise ValueError("Sum of min_weights is greater than 1. Adjust your constraints.")
        # Calculate the maximum additional weight for each asset
        max_additional = max_w - min_w
        # Generate random proportions for the remaining weight
        rand = np.random.dirichlet(np.ones(num_ativos))
        # Scale random proportions to fit the remaining weight and max_additional
        additional = np.minimum(rand * remaining, max_additional)
        # If sum(additional) < remaining, distribute the leftover to assets that can still take more
        leftover = remaining - additional.sum()
        if leftover > 1e-8:
            for i in np.argsort(max_additional - additional)[::-1]:
                add = min(leftover, max_additional[i] - additional[i])
                additional[i] += add
                leftover -= add
                if leftover <= 1e-8:
                    break
        w += additional
        weights_list.append(w)
    return np.array(weights_list)

# Simulação dos portfólios com restrições de peso
weights = generate_limited_weights(num_portfolios, num_ativos, min_weights, max_weights)
returns = np.dot(weights, retorno_anualizado)
volatility = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_carteira, weights))
risk_free_rate = 0.15
sharpe = (returns - risk_free_rate) / volatility

# Criando DataFrame dos portfólios
portfolios = pd.DataFrame({
    'Returns': returns,
    'Volatility': volatility,
    'Sharpe': sharpe
})
portfolios['Weights'] = list(weights)

# Plot
portfolios.plot(x='Volatility', y='Returns', kind='scatter', figsize=(10, 6), grid=True)
plt.title('Portfolio Returns vs Volatility')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

# Encontrando os portfólios ótimos
max_return_idx = portfolios['Returns'].idxmax()
min_volatility_idx = portfolios['Volatility'].idxmin()
max_sharpe_idx = portfolios['Sharpe'].idxmax()

def print_portfolio_info(title, portfolio, assets):
    print(f"\n{title}:")
    print(portfolio.drop('Weights'))
    print("Composição:")
    for asset, weight in zip(assets, portfolio['Weights']):
        print(f"  {asset}: {weight:.4f}")

print_portfolio_info("Portfolio com maior retorno", portfolios.loc[max_return_idx], MinhaCarteira)
print_portfolio_info("Portfolio com menor volatilidade", portfolios.loc[min_volatility_idx], MinhaCarteira)
print_portfolio_info("Portfolio com maior Sharpe Ratio", portfolios.loc[max_sharpe_idx], MinhaCarteira)