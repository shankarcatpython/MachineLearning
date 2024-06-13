import numpy as np
import matplotlib.pyplot as plt

# Given Data
initial_home_value_1994 = 129000
final_home_value_2024 = 412100
mortgage_rate = 0.03
mortgage_term_years = 30
inflation_rate = 0.025
annual_maintenance_rate = 0.03
monthly_investment = 100
annual_investment_rate = 0.03
monthly_rate = mortgage_rate / 12
n_months = mortgage_term_years * 12
years = np.arange(1994, 2025)

# Initialize arrays for yearly values
home_values = np.zeros(mortgage_term_years + 1)
maintenance_costs = np.zeros(mortgage_term_years + 1)
investment_values = np.zeros(mortgage_term_years + 1)
rent_paid = np.zeros(mortgage_term_years + 1)
mortgage_payments = np.zeros(mortgage_term_years + 1)

# Calculate yearly values
home_values[0] = initial_home_value_1994
maintenance_costs[0] = initial_home_value_1994 * annual_maintenance_rate
investment_values[0] = 0
rent_paid[0] = 0
mortgage_payments[0] = 0

monthly_mortgage_payment = (initial_home_value_1994 * monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)

for i in range(1, mortgage_term_years + 1):
    home_values[i] = initial_home_value_1994 * (1 + inflation_rate) ** i
    maintenance_costs[i] = maintenance_costs[i-1] * (1 + inflation_rate)
    investment_values[i] = investment_values[i-1] * (1 + annual_investment_rate) + 12 * monthly_investment
    rent_paid[i] = rent_paid[i-1] + 12 * 762.5  # Assuming an average monthly rent of $762.5
    mortgage_payments[i] = mortgage_payments[i-1] + 12 * monthly_mortgage_payment

# Net values for comparison
net_home_value = final_home_value_2024 - mortgage_payments[-1] - maintenance_costs[-1]
net_investment_value = investment_values[-1]

# Plotting the results
plt.figure(figsize=(14, 7))

plt.plot(years, home_values, label='Home Value Appreciation')
plt.plot(years, investment_values, label='Term Deposit Investment')
plt.plot(years, maintenance_costs, label='Maintenance Costs')
plt.plot(years, rent_paid, label='Rent Paid')
plt.plot(years, mortgage_payments, label='Mortgage Payments')

plt.title('Financial Comparison Over 30 Years')
plt.xlabel('Year')
plt.ylabel('Values in $')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the net outcomes for reference
print(f"Net Home Value (after mortgage and maintenance costs): ${net_home_value:.2f}")
print(f"Net Investment Value: ${net_investment_value:.2f}")
print(f"Total Rent Paid: ${rent_paid[-1]:.2f}")
