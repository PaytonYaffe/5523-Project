import numpy as np
import matplotlib.pyplot as plt
from dataGen import generate_data
from sdg import sgd_logistic, logistic_loss, classification_error

# ---- Experiment settings ----
sigmas   = [0.2, 0.4]
n_values = [50, 100, 500, 1000]
N_test   = 400
n_trials = 30

results = {}  # keyed by (sigma, n)

for sigma in sigmas:

    # One fixed test set per sigma value, reused across all 30 trials
    X_test, Y_test = generate_data(N_test, sigma)

    for n in n_values:
        losses = []
        errors = []

        for trial in range(n_trials):
            # Fresh training set each trial
            X_train, Y_train = generate_data(n, sigma)

            # Run SGD, get output predictor
            w = sgd_logistic(X_train, Y_train)

            # Evaluate on the held-out test set
            loss  = logistic_loss(w, X_test, Y_test)
            error = classification_error(w, X_test, Y_test)

            losses.append(loss)
            errors.append(error)

        losses = np.array(losses)
        errors = np.array(errors)

        # Compute statistics across the 30 trials
        results[(sigma, n)] = {
            'loss_mean'   : np.mean(losses),
            'loss_std'    : np.std(losses),
            'loss_min'    : np.min(losses),
            'excess_risk' : np.mean(losses) - np.min(losses),
            'error_mean'  : np.mean(errors),
            'error_std'   : np.std(errors),
        }

        print(f"sigma={sigma} | n={n:4d} | "
              f"Loss Mean={results[(sigma,n)]['loss_mean']:.4f} | "
              f"Loss Std={results[(sigma,n)]['loss_std']:.4f} | "
              f"Loss Min={results[(sigma,n)]['loss_min']:.4f} | "
              f"Excess Risk={results[(sigma,n)]['excess_risk']:.4f} | "
              f"Error Mean={results[(sigma,n)]['error_mean']:.4f} | "
              f"Error Std={results[(sigma,n)]['error_std']:.4f}")

# ---- Print full results table ----
print("\n--- Full Results Table ---")
print(f"{'sigma':>6} {'n':>6} {'N':>5} {'Trials':>7} {'Loss Mean':>10} {'Loss Std':>9} {'Loss Min':>9} {'Excess Risk':>12} {'Error Mean':>11} {'Error Std':>10}")
for sigma in sigmas:
    for n in n_values:
        r = results[(sigma, n)]
        print(f"{sigma:>6} {n:>6} {N_test:>5} {n_trials:>7} "
              f"{r['loss_mean']:>10.4f} {r['loss_std']:>9.4f} {r['loss_min']:>9.4f} "
              f"{r['excess_risk']:>12.4f} {r['error_mean']:>11.4f} {r['error_std']:>10.4f}")

# ---- Plot 1: Excess Risk vs n ----
plt.figure(figsize=(7, 5))
for sigma in sigmas:
    excess_risks = [results[(sigma, n)]['excess_risk'] for n in n_values]
    loss_stds    = [results[(sigma, n)]['loss_std']    for n in n_values]
    plt.errorbar(n_values, excess_risks, yerr=loss_stds,
                 marker='o', capsize=5, label=f'σ = {sigma}')

plt.title("Expected Excess Risk vs. Number of Training Examples")
plt.xlabel("n (number of training examples)")
plt.ylabel("Excess Risk (mean − min)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("excess_risk_plot.png", dpi=150)
plt.show()
print("Saved: excess_risk_plot.png")

# ---- Plot 2: Classification Error vs n ----
plt.figure(figsize=(7, 5))
for sigma in sigmas:
    error_means = [results[(sigma, n)]['error_mean'] for n in n_values]
    error_stds  = [results[(sigma, n)]['error_std']  for n in n_values]
    plt.errorbar(n_values, error_means, yerr=error_stds,
                 marker='s', capsize=5, label=f'σ = {sigma}')

plt.title("Expected Classification Error vs. Number of Training Examples")
plt.xlabel("n (number of training examples)")
plt.ylabel("Classification Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("classification_error_plot.png", dpi=150)
plt.show()
print("Saved: classification_error_plot.png")
