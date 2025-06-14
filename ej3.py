import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

alpha_0 = 2
beta_0 = 3

mean_prior = alpha_0 / (alpha_0 + beta_0)
variance_prior = (alpha_0 * beta_0) / \
    ((alpha_0 + beta_0)**2 * (alpha_0 + beta_0 + 1))

print(f"--- Distribucin a Priori ---")
print(f"Parametros a priori (alpha_0, beta_0): ({alpha_0}, {beta_0})")
print(f"Media a priori: {mean_prior:.4f}")
print(f"Varianza a priori: {variance_prior:.6f}")
print("-" * 40)

n_clientes = 8
k_morosos = 3

print(f"--- Datos Observados ---")
print(f"Numero total de clientes (n): {n_clientes}")
print(f"Numero de clientes morosos (k): {k_morosos}")
print("-" * 40)

alpha_prime = alpha_0 + k_morosos
beta_prime = beta_0 + (n_clientes - k_morosos)

print(f"--- Distribucion a Posteriori ---")
print(
    f"Nuevos parametros a posteriori (alpha', beta'): ({alpha_prime}, {beta_prime})")

mean_posterior = alpha_prime / (alpha_prime + beta_prime)
print(
    f"Media a posteriori (E[p|datos]): {mean_posterior:.4f} (aprox. {mean_posterior*100:.2f}%)")

variance_posterior = (alpha_prime * beta_prime) / \
    ((alpha_prime + beta_prime)**2 * (alpha_prime + beta_prime + 1))
print(f"Varianza a posteriori (Var[p|datos]): {variance_posterior:.6f}")
print("-" * 40)

p_values = np.linspace(0, 1, 500)

pdf_prior = stats.beta.pdf(p_values, alpha_0, beta_0)

pdf_posterior = stats.beta.pdf(p_values, alpha_prime, beta_prime)

plt.figure(figsize=(10, 6))
plt.plot(p_values, pdf_prior,
         label=f'A Priori Beta({alpha_0}, {beta_0})', linestyle='--', color='blue')
plt.plot(p_values, pdf_posterior,
         label=f'A Posteriori Beta({alpha_prime}, {beta_prime})', color='red')
plt.axvline(mean_prior, color='blue', linestyle=':',
            label=f'Media a Priori ({mean_prior:.2f})')
plt.axvline(mean_posterior, color='red', linestyle=':',
            label=f'Media a Posteriori ({mean_posterior:.2f})')

plt.title('Distribucion A Priori y A Posteriori del Porcentaje de Morosidad')
plt.xlabel('Porcentaje de Morosidad (p)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
