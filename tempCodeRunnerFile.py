# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Data
# data = {
#     'Model': ['Multivariate Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Ensemble Learning'],
#     'Accuracy': [89, 92, 93, 96, 96],
#     'Computation Time': [4, 5.5, 7, 10, 15]
# }

# df = pd.DataFrame(data)

# # Set the aesthetic style of the plots
# sns.set(style="whitegrid")

# # Create the line plot
# plt.figure(figsize=(10, 6))
# lineplot = sns.lineplot(data=df, x='Computation Time', y='Accuracy', marker='o')

# # Add annotations for each point
# for i in range(len(df)):
#     lineplot.text(df['Computation Time'][i], df['Accuracy'][i], df['Model'][i], horizontalalignment='left', size='medium', color='black', weight='semibold')

# # Adding title and labels
# plt.title('Accuracy vs Computation Time for ML Models')
# plt.xlabel('Computation Time (min)')
# plt.ylabel('Accuracy (%)')

# # Adding metadata as text
# metadata = "Processor: Intel Core i5 @ 3.3 GHz, 16GB RAM, 4GB GPU"
# plt.figtext(0.5, -0.1, metadata, wrap=True, horizontalalignment='center', fontsize=12)

# # Show the plot
# plt.tight_layout()
# plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Model': ['Multivariate Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Ensemble Learning'],
    'Accuracy': [89, 92, 93, 96, 96],
    'Computation Time': [4, 5.5, 7, 10, 15]
}

df = pd.DataFrame(data)

# Set the aesthetic style of the plots
sns.set(style="darkgrid")

# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot accuracy on the first y-axis
sns.lineplot(data=df, x='Model', y='Accuracy', marker='o', ax=ax1, label="Accuracy", color='b')
ax1.set_ylabel('Accuracy (%)', color='b')
ax1.set_xlabel('Model')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for computation time
ax2 = ax1.twinx()
sns.lineplot(data=df, x='Model', y='Computation Time', marker='o', ax=ax2, label="Computation Time", color='r')
ax2.set_ylabel('Computation Time (min)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding title
plt.title('Accuracy and Computation Time for ML Models on GPU node ')

# Adding metadata as text
metadata = "Processor: Intel Core i5 @ 3.3 GHz, 16GB RAM, 4GB GPU"
plt.figtext(0.5, -0.1, metadata, wrap=True, horizontalalignment='center', fontsize=12)

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()
