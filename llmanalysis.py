import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Model Name": ["Gemini Pro", "GPT-4", "GPT-3", "Codex", "Claude", "LLaMA 2", "MPT", "BERT", "T5", "LaMDA", "BlenderBot", "GPT-Neo", "GPT-J", "BLOOM", "Megatron-Turing NLG", "Siri Language Model", "CodeGen", "Dolly"],
    "Open Source": ["No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes"],
    "Company": ["Google", "OpenAI", "OpenAI", "OpenAI", "Anthropic", "Meta", "MosaicML", "Google", "Google", "Google", "Meta AI", "EleutherAI", "EleutherAI", "BigScience", "NVIDIA/Microsoft", "Apple", "Salesforce", "Databricks"],
    "Parameters": [1e6, None, 175e9, 12e9, 52e9, 65e9, 6.7e9, 110e6, 220e6, 137e9, 2.7e9, 2.7e9, 6e9, 176e9, 530e9, None, 16e9, 12e9],
    "Context Window": [1e6, 8192, 2048, 2048, 8192, 4096, 2048, 512, 512, None, None, 2048, 2048, 2048, 1024, None, 2048, None],
    "Major Use Case": ["Text generation, code generation, analysis", "General-purpose text generation", "General-purpose text generation", "Code generation", "Safe and aligned language generation", "Text generation, code generation, research", "General-purpose text generation", "Text classification and NER", "Text translation and summarization", "Conversational AI", "Conversational AI", "General-purpose text generation", "General-purpose text generation", "General-purpose text generation", "General-purpose text generation", "Conversational AI for Siri", "Code generation", "General-purpose text generation"],
    "Cost per 10k Requests": ["Contact Google", "$0.03 - $0.12", "$0.02", "$0.10", "Varies", "Free", "Free", "Free", "Free", "Not publicly available", "Free", "Free", "Free", "Free", "Not publicly available", "Not publicly available", "Free", "Free"],
    "Constraints": ["Proprietary model", "Token limit", "Closed-source", "Closed-source", "Limited availability, closed-source", "Requires powerful hardware", "Computationally expensive", "Requires fine-tuning", "Requires fine-tuning", "Closed-source", "Research purposes", "Computationally expensive", "Computationally expensive", "Requires large resources", "Closed-source", "Closed-source", "Requires fine-tuning", "Requires large resources"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Replace None with NaN for proper handling
df['Parameters'] = df['Parameters'].replace({None: np.nan})
df['Context Window'] = df['Context Window'].replace({None: np.nan})

# Plot: Number of Parameters for Each Model (Log Scale)
plt.figure(figsize=(14, 8))
plt.barh(df['Model Name'], df['Parameters'], color='skyblue')
plt.xscale('log')
plt.xlabel('Number of Parameters (Log Scale)')
plt.title('Number of Parameters in Different AI Models')
plt.grid(True, which="both", ls="--")
plt.annotate('Note: Log scale used for better visualization', xy=(1e6, 0), xytext=(1e8, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# Plot: Context Window Size (Log Scale)
plt.figure(figsize=(14, 8))
plt.barh(df['Model Name'], df['Context Window'], color='lightgreen')
plt.xscale('log')
plt.xlabel('Context Window Size (Log Scale)')
plt.title('Context Window Size in Different AI Models')
plt.grid(True, which="both", ls="--")
plt.annotate('Note: Log scale used for better visualization', xy=(512, 0), xytext=(4096, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# Pie Chart: Open Source vs Closed Source
open_source_counts = df['Open Source'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(open_source_counts, labels=open_source_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
plt.title('Open Source vs Closed Source Models')
plt.show()

# Scatter Plot: Parameters vs Context Window Size (Log Scale)
plt.figure(figsize=(14, 8))
plt.scatter(df['Context Window'], df['Parameters'], color='purple')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Context Window Size (Log Scale)')
plt.ylabel('Number of Parameters (Log Scale)')
plt.title('Parameters vs Context Window Size')
plt.grid(True, which="both", ls="--")
plt.annotate('Note: Log scale used for better visualization', xy=(512, 1e6), xytext=(4096, 1e9),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# Bar Plot: Cost per 10k Requests (filtering only numerical values for clear plotting)
df_cost = df[df['Cost per 10k Requests'].str.contains('\$')].copy()
df_cost['Cost per 10k Requests (numeric)'] = df_cost['Cost per 10k Requests'].str.extract(r'(\d+\.?\d*)').astype(float)
plt.figure(figsize=(14, 8))
plt.bar(df_cost['Model Name'], df_cost['Cost per 10k Requests (numeric)'], color='orange')
plt.ylabel('Cost per 10k Requests ($)')
plt.title('Cost per 10k Requests for Different AI Models')
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
