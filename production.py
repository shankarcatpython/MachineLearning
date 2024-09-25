import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from faker import Faker
import plotly.express as px
import plotly.graph_objects as go

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
Faker.seed(42)

# Function to create industry-specific synthetic time series data
def create_industry_synthetic_data(start_time, end_time, freq, num_records, host_names):
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    num_hosts = len(host_names)
    
    # CPU Data
    cpu_data = pd.DataFrame({
        'Time': np.tile(timestamps, num_hosts),
        'Host': np.repeat(host_names, len(timestamps)),
        'CPU_Usage': np.random.uniform(10, 90, num_records),
        'User_Time': np.random.uniform(5, 50, num_records),
        'System_Time': np.random.uniform(5, 40, num_records),
        'Idle_Time': np.random.uniform(0, 50, num_records),
        'IOWait': np.random.uniform(0, 10, num_records)
    })
    
    # Memory Data
    memory_data = pd.DataFrame({
        'Time': np.tile(timestamps, num_hosts),
        'Host': np.repeat(host_names, len(timestamps)),
        'Total_Memory': np.random.randint(8000, 32000, num_records),
        'Used_Memory': np.random.randint(2000, 25000, num_records),
        'Free_Memory': np.random.randint(500, 8000, num_records),
        'Cache_Memory': np.random.randint(500, 10000, num_records),
        'Swap_Used': np.random.randint(0, 2000, num_records)
    })
    
    # App Stats Data
    app_stats_data = pd.DataFrame({
        'Time': np.tile(timestamps, num_hosts),
        'Host': np.repeat(host_names, len(timestamps)),
        'Request_Count': np.random.randint(100, 5000, num_records),
        'Error_Count': np.random.randint(0, 50, num_records),
        'Latency': np.random.uniform(100, 500, num_records),  # Latency in milliseconds
        'Throughput': np.random.uniform(50, 200, num_records)  # Requests per second
    })
    
    return cpu_data, memory_data, app_stats_data

# Parameters
start_time = datetime.now() - timedelta(days=1)
end_time = datetime.now()
freq = '15min'  # Updated frequency to avoid deprecation warning
host_names = [fake.hostname() for _ in range(5)]

# Create industry-specific synthetic data
cpu_data, memory_data, app_stats_data = create_industry_synthetic_data(
    start_time, 
    end_time, 
    freq, 
    len(pd.date_range(start=start_time, end=end_time, freq=freq)) * len(host_names), 
    host_names
)

# Function to aggregate data based on custom time format
def aggregate_by_time(df, time_format='%Y%m%d%H%M'):
    df['Time_Aggregated'] = df['Time'].dt.strftime(time_format)
    # Drop non-numeric columns before aggregation
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_agg = df[['Time_Aggregated', 'Host'] + list(numeric_cols)]
    # Sum up numeric columns grouped by Time_Aggregated and Host
    aggregated_df = df_agg.groupby(['Time_Aggregated', 'Host']).sum().reset_index()
    return aggregated_df

# Aggregate the data based on YYYYMMDDHHMM and host
cpu_aggregated = aggregate_by_time(cpu_data)
memory_aggregated = aggregate_by_time(memory_data)
app_stats_aggregated = aggregate_by_time(app_stats_data)

# Merge the aggregated data
merged_aggregated = pd.merge(cpu_aggregated, memory_aggregated, on=['Time_Aggregated', 'Host'], how='outer')
merged_aggregated = pd.merge(merged_aggregated, app_stats_aggregated, on=['Time_Aggregated', 'Host'], how='outer')

# Feature Engineering
merged_aggregated['Used_Memory_Percent'] = (merged_aggregated['Used_Memory'] / merged_aggregated['Total_Memory']) * 100
merged_aggregated['Error_Rate'] = (merged_aggregated['Error_Count'] / merged_aggregated['Request_Count']) * 100

# Fill missing values using forward fill and backward fill
merged_aggregated.ffill(inplace=True)
merged_aggregated.bfill(inplace=True)

# Apply Anomaly Detection
features = ['CPU_Usage', 'Used_Memory_Percent', 'Error_Rate']
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
merged_aggregated['Anomaly_Score'] = isolation_forest.fit_predict(merged_aggregated[features])
merged_aggregated['Anomaly'] = merged_aggregated['Anomaly_Score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Calculate baseline metrics for normal data
normal_data = merged_aggregated[merged_aggregated['Anomaly'] == 'Normal']
baseline_metrics = {
    'CPU_Usage': {
        'mean': normal_data['CPU_Usage'].mean(),
        'std': normal_data['CPU_Usage'].std()
    },
    'Used_Memory_Percent': {
        'mean': normal_data['Used_Memory_Percent'].mean(),
        'std': normal_data['Used_Memory_Percent'].std()
    },
    'Error_Rate': {
        'mean': normal_data['Error_Rate'].mean(),
        'std': normal_data['Error_Rate'].std()
    }
}

# Identify and Explain Anomalies with detailed comparison to baseline
anomalies = merged_aggregated[merged_aggregated['Anomaly'] == 'Anomaly']
anomaly_reasons = []

for index, row in anomalies.iterrows():
    reason = []
    
    # Compare CPU Usage to baseline
    cpu_diff = row['CPU_Usage'] - baseline_metrics['CPU_Usage']['mean']
    if cpu_diff > baseline_metrics['CPU_Usage']['std']:
        reason.append(f"CPU Usage is {cpu_diff:.2f} units higher than average ({baseline_metrics['CPU_Usage']['mean']:.2f})")

    # Compare Used Memory Percent to baseline
    mem_diff = row['Used_Memory_Percent'] - baseline_metrics['Used_Memory_Percent']['mean']
    if mem_diff > baseline_metrics['Used_Memory_Percent']['std']:
        reason.append(f"Memory Usage is {mem_diff:.2f} units higher than average ({baseline_metrics['Used_Memory_Percent']['mean']:.2f})")
        
    # Compare Error Rate to baseline
    err_diff = row['Error_Rate'] - baseline_metrics['Error_Rate']['mean']
    if err_diff > baseline_metrics['Error_Rate']['std']:
        reason.append(f"Error Rate is {err_diff:.2f} units higher than average ({baseline_metrics['Error_Rate']['mean']:.2f})")

    # Append detailed reason with time and host
    anomalies.loc[index, 'Anomaly_Reason'] = f"Time: {row['Time_Aggregated']}, Host: {row['Host']}, Reason: {', '.join(reason)}"

# Display the data and anomalies
print("Sample of Merged Aggregated Data:")
print(merged_aggregated.head())
print("\nAnomalies with Host and Detailed Reasons:")
print(anomalies[['Time_Aggregated', 'Host', 'Anomaly_Reason']])

# Plotting the anomalies using Plotly

# CPU Usage Anomalies
fig_cpu = px.scatter(merged_aggregated, x='Time_Aggregated', y='CPU_Usage', color='Anomaly', 
                     symbol='Host', title='CPU Usage Anomalies Over Time by Host',
                     labels={'CPU_Usage': 'CPU Usage (%)', 'Time_Aggregated': 'Time'})
fig_cpu.update_layout(xaxis=dict(tickangle=45))
fig_cpu.show()

# Memory Usage Anomalies
fig_memory = px.scatter(merged_aggregated, x='Time_Aggregated', y='Used_Memory_Percent', color='Anomaly',
                        symbol='Host', title='Memory Usage Anomalies Over Time by Host',
                        labels={'Used_Memory_Percent': 'Used Memory (%)', 'Time_Aggregated': 'Time'})
fig_memory.update_layout(xaxis=dict(tickangle=45))
fig_memory.show()

# Error Rate Anomalies
fig_error = px.scatter(merged_aggregated, x='Time_Aggregated', y='Error_Rate', color='Anomaly',
                       symbol='Host', title='Application Error Rate Anomalies Over Time by Host',
                       labels={'Error_Rate': 'Error Rate (%)', 'Time_Aggregated': 'Time'})
fig_error.update_layout(xaxis=dict(tickangle=45))
fig_error.show()
