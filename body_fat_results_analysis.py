import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("body_fat_results.csv")
alphas = data.groupby('alpha').agg({'mse': 'min'}).reset_index()

plt.scatter(alphas['alpha'], alphas['mse'])
plt.title("Mean Squared Error across Alpha Values with Optimal Lambda")
plt.xlabel("Alpha")
plt.ylabel("Mean Squared Error")
plt.savefig("elastic_net_results_plots/mse_vs_alpha.png")
plt.close("all")

lambdas = data.groupby('lambda').agg({'mse': 'min'}).reset_index()

plt.scatter(np.log10(lambdas['lambda']), lambdas['mse'])
plt.title("Mean Squared Error across Lambda Values with Optimal Alpha")
plt.xlabel("Log Lambda")
plt.ylabel("Mean Squared Error")
plt.savefig("elastic_net_results_plots/mse_vs_lambda.png")
plt.close("all")

best_lambdas = []
best_coefs = []
for t in alphas.itertuples():
    data_row = data[(data['alpha'] == t.alpha) & (data['mse'] == t.mse)].iloc[0]
    best_lambdas.append(data_row['lambda'])
    best_coefs.append([data_row['intercept'], data_row['x1'], data_row['x2'], data_row['x3']])

plt.scatter(alphas['alpha'], np.log10(best_lambdas))
plt.xlabel("Alpha")
plt.ylabel("Log Optimal Lambda")
plt.title("Optimal Lambda Values for Alpha Values")
plt.tight_layout()
plt.savefig("elastic_net_results_plots/optimal_lambda_vs_alpha.png")
plt.close("all")

best_coefs = np.array(best_coefs).T

plt.scatter(alphas['alpha'], best_coefs[0], label="Intercept")
for i in range(1, 4):
    plt.scatter(alphas['alpha'], best_coefs[i], label=f"x{i}")
plt.legend()
plt.title("Optimal Parameter Values across Alpha Values with Optimal Lambda")
plt.ylabel("Optimal Parameters")
plt.xlabel("Alpha")
plt.savefig("elastic_net_results_plots/predictor_coefficients_vs_alpha.png")
plt.close("all")

best_alphas = []
best_coefs = []
for t in lambdas.itertuples():
    data_row = data[(data['lambda'] == t._1) & (data['mse'] == t.mse)].iloc[0]
    best_alphas.append(data_row['alpha'])
    best_coefs.append([data_row['intercept'], data_row['x1'], data_row['x2'], data_row['x3']])

plt.scatter(np.log10(lambdas['lambda']), best_alphas)
plt.xlabel("Log Lambda")
plt.ylabel("Optimal Alpha")
plt.title("Optimal Alpha Values for Lambda Values")
plt.tight_layout()
plt.savefig("elastic_net_results_plots/optimal_alpha_vs_lambda.png")
plt.close("all")

best_coefs = np.array(best_coefs).T
plt.scatter(np.log10(lambdas['lambda']), best_coefs[0], label="Intercept")
for i in range(1, 4):
    plt.scatter(np.log10(lambdas['lambda']), best_coefs[i], label=f"x{i}")
plt.legend()
plt.title("Optimal Parameter Values across Lambda Values with Optimal Alpha")
plt.ylabel("Optimal Parameters")
plt.xlabel("Log Lambda")
plt.savefig("elastic_net_results_plots/predictor_coefficients_vs_lambda.png")
plt.close("all")

