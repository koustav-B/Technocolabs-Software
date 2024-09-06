import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence
import shap
import tkinter as tk
from tkinter import messagebox

# Ask the user to input the path of the CSV file
file_path = input("Enter the path of your CSV file: ")

# Load the CSV file
df = pd.read_csv(file_path, encoding='utf-8')

# Data Cleaning and Preparation
df_cleaned = df.copy()
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].median())
df_cleaned['english.grade'] = df_cleaned['english.grade'].fillna(df_cleaned['english.grade'].mean())
df_cleaned['city'] = df_cleaned['city'].fillna(df_cleaned['city'].mode()[0])
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')
df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')
df_cleaned['gender'] = df_cleaned['gender'].str.strip().str.lower()
df_cleaned['nationality'] = df_cleaned['nationality'].str.strip().str.title()
df_cleaned['city'] = df_cleaned['city'].str.strip().str.title()
df_cleaned = df_cleaned[(df_cleaned['age'] >= 5) & (df_cleaned['age'] <= 100)]
df_cleaned = df_cleaned.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

# Scaling numerical values
scaler = MinMaxScaler()
df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']] = scaler.fit_transform(
    df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']]
)

# Feature Selection and Engineering
features = ['age', 'english.grade', 'math.grade', 'sciences.grade', 'language.grade']
target = 'portfolio_rating'  # Example target variable

# Check if target variable exists
if target in df_cleaned.columns:
    X = df_cleaned[features]
    y = df_cleaned[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training and Evaluation
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(),
        'Support Vector Regression': SVR()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'R² Score': r2}

        # Permutation Importance
        if hasattr(model, 'feature_importances_'):
            importances = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
            results[name]['Permutation Importance'] = importances.importances_mean

        # Partial Dependence Plots
        if hasattr(model, 'predict'):
            fig, ax = plt.subplots(figsize=(10, 6))
            partial_dependence.plot_partial_dependence(model, X_train_scaled, features, ax=ax)
            plt.title(f'Partial Dependence Plots for {name}')
            plt.show()

        # SHAP Values
        if hasattr(model, 'predict'):
            explainer = shap.Explainer(model, X_train_scaled)
            shap_values = explainer(X_test_scaled)
            shap.summary_plot(shap_values, X_test_scaled, feature_names=features)

    # Display results in a GUI
    def show_results():
        results_text = ""
        for model_name, metrics in results.items():
            results_text += f"{model_name}:\n"
            results_text += f"  Mean Squared Error: {metrics['MSE']:.2f}\n"
            results_text += f"  R² Score: {metrics['R² Score']:.2f}\n"
            if 'Permutation Importance' in metrics:
                results_text += f"  Permutation Importance: {metrics['Permutation Importance']}\n"
            results_text += "\n"

        # Create the main window
        root = tk.Tk()
        root.title("Machine Learning Results")

        # Create a Text widget to display results
        text_widget = tk.Text(root, wrap=tk.WORD, height=15, width=60)
        text_widget.pack(padx=10, pady=10)
        text_widget.insert(tk.END, results_text)

        # Add a Quit button
        quit_button = tk.Button(root, text="Quit", command=root.quit)
        quit_button.pack(pady=5)

        # Run the GUI
        root.mainloop()

    # Show results
    show_results()

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_combined_file.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned data saved to: {cleaned_file_path}")
