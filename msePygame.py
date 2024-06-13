import matplotlib.pyplot as plt
import numpy as np

# Generate random values
def generate_random_values(n=10):
    actual = np.random.randint(100, 201, n)
    predicted = np.random.randint(100, 201, n)
    return actual, predicted

# Calculate losses
def calculate_losses(actual, predicted, quantile=0.5, delta=1.0):
    mse_individual = (actual - predicted) ** 2
    mae_individual = np.abs(actual - predicted)
    huber_individual = np.where(np.abs(actual - predicted) <= delta, 
                                0.5 * (actual - predicted) ** 2, 
                                delta * (np.abs(actual - predicted) - 0.5 * delta))
    log_cosh_individual = np.log(np.cosh(predicted - actual))
    quantile_loss_individual = np.where(actual >= predicted, 
                                        quantile * (actual - predicted), 
                                        (1 - quantile) * (predicted - actual))
    
    mse = np.mean(mse_individual)
    mae = np.mean(mae_individual)
    huber = np.mean(huber_individual)
    log_cosh = np.mean(log_cosh_individual)
    quantile_loss = np.mean(quantile_loss_individual)
    
    return mse, mae, huber, log_cosh, quantile_loss, mse_individual, mae_individual, huber_individual, log_cosh_individual, quantile_loss_individual

# Generate and plot losses
def plot_losses():
    actual, predicted = generate_random_values()
    mse, mae, huber, log_cosh, quantile_loss, mse_individual, mae_individual, huber_individual, log_cosh_individual, quantile_loss_individual = calculate_losses(actual, predicted)
    
    loss_names = ['MSE', 'MAE', 'Huber', 'Log-Cosh', 'Quantile']
    loss_individual = [mse_individual, mae_individual, huber_individual, log_cosh_individual, quantile_loss_individual]
    loss_mean = [mse, mae, huber, log_cosh, quantile_loss]
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    for ax, loss_name, loss_ind, loss_m in zip(axs, loss_names, loss_individual, loss_mean):
        ax.plot(loss_ind, label=f'{loss_name} Individual')
        ax.axhline(y=loss_m, color='r', linestyle='--', label=f'Mean {loss_name}')
        ax.set_title(f'{loss_name} Loss')
        ax.legend()
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Loss Value')

    plt.tight_layout()
    plt.show()

# Plot the losses
plot_losses()
