import numpy as np
import pandas as pd
import statsmodels.api as sm

# 
file_path = "C:/Users/qaz99/OneDrive/桌面/python/BM25/25bmms.csv"
data = pd.read_csv(file_path)

#
dates = data['date']
SENTS = data['SENT']
asset_returns = data[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25']].to_numpy()
market_returns = data['Market'].to_numpy()
markets = data['Market']

# 
weight_columns = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25']
weights_data = data[weight_columns].values
    
# 
mw_weight_columns = ['MW1', 'MW2', 'MW3', 'MW4', 'MW5', 'MW6', 'MW7', 'MW8', 'MW9', 'MW10', 'MW11', 'MW12', 'MW13', 'MW14', 'MW15', 'MW16', 'MW17', 'MW18', 'MW19', 'MW20', 'MW21', 'MW22', 'MW23', 'MW24', 'MW25']
mw_weights_data = data[mw_weight_columns].values

# 
def markowitz_weights(cov_matrix, mean_returns, gamma=2.5):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    numerator = inv_cov_matrix @ mean_returns
    denominator = np.ones(len(mean_returns)).T @ numerator
    weights = numerator / denominator
    return weights

# 
def new_mu_e(cov_matrix, weights, gamma_m=2.5):
    return gamma_m * np.dot(cov_matrix, weights)

#
def bl_extended_mu_T(cov_matrix, tau, mu_bl, sample_mean_returns, sample_cov_matrix, T):
    delta = tau * cov_matrix
    delta_inv = np.linalg.inv(delta)
    sample_cov_matrix_T_inv = np.linalg.inv(sample_cov_matrix / T)
    mu_T = np.linalg.inv(delta_inv + sample_cov_matrix_T_inv) @ (delta_inv @ mu_bl + sample_cov_matrix_T_inv @ sample_mean_returns)
    return mu_T

def bl_extended_sigma_T(cov_matrix, tau, sample_cov_matrix, T):
    delta = tau * cov_matrix
    delta_inv = np.linalg.inv(delta)
    sample_cov_matrix_T_inv = np.linalg.inv(sample_cov_matrix / T)
    sigma_T = cov_matrix + np.linalg.inv(delta_inv + sample_cov_matrix_T_inv)
    return sigma_T

#期間
window_size = 120
T = window_size  # 
initial_tau = 0.05
initial_tau1 = 1/T  # 

# 
results = []

# 
for start in range(len(asset_returns) - window_size):
    end = start + window_size
    window_market_returns = market_returns[start:end]
    window_asset_returns = asset_returns[start:end]
    # 
    if SENTS[end - 1] < 0:
        tau = initial_tau ** (1-SENTS)
        tau1 = initial_tau1 ** (1-SENTS)
    else:
        tau = initial_tau ** (SENTS+1)
        tau1 = initial_tau ** (SENTS+1)
    

    market_mean = np.mean(window_market_returns)
    market_var = np.var(window_market_returns)
    assets_mean = np.mean(window_asset_returns, axis=0)

    #  beta 和 alpha 
    betas = []
    alphas = []
    residuals = []
    
    # CAPM參數假設
    for i in range(window_asset_returns.shape[1]):
        X = sm.add_constant(window_market_returns)
        Y = window_asset_returns[:, i]
        model = sm.OLS(Y, X).fit()
        alphas.append(model.params[0])
        betas.append(model.params[1])
        residuals.append(np.var(model.resid))
    
    alphas = np.array(alphas)
    betas = np.array(betas)
    residuals = np.array(residuals)
    
    expected_returns = betas * market_mean
    
    cov_matrix = np.outer(betas, betas) * market_var + np.diag(residuals)
    
    adjusted_cov_matrix = cov_matrix * (1 + tau)
    
    cov_matrix1 = np.cov(window_asset_returns.T)
    adjusted_cov_matrix1 = cov_matrix1 * (1 + tau)

    weights_original = markowitz_weights(cov_matrix, expected_returns)
    
    weights_data_current = weights_data[start + window_size - 1, :].reshape(-1, 1)
    mu_e_new1 = new_mu_e(cov_matrix, weights_data_current, gamma_m=2.5)
    
    mw_weights_data_current = mw_weights_data[start + window_size - 1, :].reshape(-1, 1)
    mu_e_new2 = new_mu_e(cov_matrix, mw_weights_data_current, gamma_m=2.5)
    
    weights_new1 = weights_data_current.flatten()
    
    weights_new2 = mw_weights_data_current.flatten()
    
    mu_T1 = bl_extended_mu_T(cov_matrix, tau, mu_e_new1.flatten(), expected_returns, cov_matrix, T)
    mu_T2 = bl_extended_mu_T(cov_matrix, tau, mu_e_new2.flatten(), expected_returns, cov_matrix, T)
    mu_T3 = bl_extended_mu_T(cov_matrix, tau1, mu_e_new1.flatten(), expected_returns, cov_matrix, T)
    mu_T4 = bl_extended_mu_T(cov_matrix, tau1, mu_e_new2.flatten(), expected_returns, cov_matrix, T)
    mu_T5 = bl_extended_mu_T(cov_matrix1, tau, mu_e_new1.flatten(), assets_mean, cov_matrix1, T)
    mu_T6 = bl_extended_mu_T(cov_matrix1, tau, mu_e_new2.flatten(), assets_mean, cov_matrix1, T)
    mu_T7 = bl_extended_mu_T(cov_matrix1, tau1, mu_e_new1.flatten(), assets_mean, cov_matrix1, T)
    mu_T8 = bl_extended_mu_T(cov_matrix1, tau1, mu_e_new2.flatten(), assets_mean, cov_matrix1, T)
    
    sigma_T1 = bl_extended_sigma_T(cov_matrix, tau, cov_matrix, T)
    sigma_T2 = bl_extended_sigma_T(cov_matrix1, tau, cov_matrix1, T)

    weights_bl0 = markowitz_weights(cov_matrix1, assets_mean, gamma=2.5)
    
    weights_bl1 = markowitz_weights(sigma_T1, mu_T1, gamma=2.5)
    
    weights_bl2 = markowitz_weights(sigma_T1, mu_T2, gamma=2.5)
    
    weights_bl3 = markowitz_weights(sigma_T2, mu_T3, gamma=2.5)
   
    weights_bl4 = markowitz_weights(sigma_T2, mu_T4, gamma=2.5)

    weights_bl5 = markowitz_weights(sigma_T1, mu_T5, gamma=2.5)
    

    weights_bl6 = markowitz_weights(sigma_T1, mu_T6, gamma=2.5)

    weights_bl7 = markowitz_weights(sigma_T2, mu_T7, gamma=2.5)
    

    weights_bl8 = markowitz_weights(sigma_T2, mu_T8, gamma=2.5)
    
    # 個別參數投資組合報酬率
    if start + window_size < len(asset_returns):
        test_return = asset_returns[start + window_size, :]
        portfolio_return_original = np.dot(weights_original.flatten(), test_return)
        portfolio_return_new1 = np.dot(weights_new1, test_return)
        portfolio_return_new2 = np.dot(weights_new2, test_return)
        portfolio_return_bl0 = np.dot(weights_bl0.flatten(), test_return)
        portfolio_return_bl1 = np.dot(weights_bl1.flatten(), test_return)
        portfolio_return_bl2 = np.dot(weights_bl2.flatten(), test_return)
        portfolio_return_bl3 = np.dot(weights_bl3.flatten(), test_return)
        portfolio_return_bl4 = np.dot(weights_bl4.flatten(), test_return)
        portfolio_return_bl5 = np.dot(weights_bl5.flatten(), test_return)
        portfolio_return_bl6 = np.dot(weights_bl6.flatten(), test_return)
        portfolio_return_bl7 = np.dot(weights_bl7.flatten(), test_return)
        portfolio_return_bl8 = np.dot(weights_bl8.flatten(), test_return)
        test_date = dates[start + window_size]
        SENT = SENTS[start + window_size]
        Market = markets[start + window_size]
        
        # 結果
        result = {
            'Date': test_date,
            'SENT': SENT,
            'Market': Market,
            'Portfolio Return Original': portfolio_return_original,
            'Portfolio Return New1': portfolio_return_new1,
            'Portfolio Return New2': portfolio_return_new2,
            'Portfolio Return BL0': portfolio_return_bl0,
            'Portfolio Return BL1': portfolio_return_bl1,
            'Portfolio Return BL2': portfolio_return_bl2,
            'Portfolio Return BL3': portfolio_return_bl3,
            'Portfolio Return BL4': portfolio_return_bl4,
            'Portfolio Return BL5': portfolio_return_bl5,
            'Portfolio Return BL6': portfolio_return_bl6,
            'Portfolio Return BL7': portfolio_return_bl7,
            'Portfolio Return BL8': portfolio_return_bl8
         }
      
        results.append(result)


results_df = pd.DataFrame(results)


output_file_path = r"C:\Users\qaz99\OneDrive\桌面\python\BM25\25BMFinalN12.csv"
results_df.to_csv(output_file_path, index=False)

print(f"结果已保存到 {output_file_path}")
