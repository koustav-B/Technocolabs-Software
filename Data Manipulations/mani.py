import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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

# Data Manipulations

# 1. Grouping and Aggregating
grouped_data = df_cleaned.groupby('gender').agg({
    'age': 'mean',
    'english.grade': 'mean',
    'math.grade': 'mean',
    'sciences.grade': 'mean',
    'language.grade': 'mean'
}).reset_index()
print("\nGrouped Data (Mean Grades and Age by Gender):")
print(grouped_data)

# 2. Sorting
sorted_data = df_cleaned.sort_values(by=['english.grade', 'math.grade'], ascending=[False, False])
print("\nTop 5 Records Sorted by English and Math Grades:")
print(sorted_data.head())

# 3. Renaming Columns
df_cleaned = df_cleaned.rename(columns={'portfolio.rating': 'portfolio_rating', 'coverletter.rating': 'coverletter_rating', 'refletter.rating': 'refletter_rating'})
print("\nData with Renamed Columns:")
print(df_cleaned.head())

# 4. Combining DataFrames
# For demonstration, creating another DataFrame with some additional information
additional_info = pd.DataFrame({
    'id': df_cleaned['id'],
    'extra_info': ['Info' + str(i) for i in range(len(df_cleaned))]
})

# Merging with the cleaned DataFrame
df_combined = pd.merge(df_cleaned, additional_info, on='id', how='left')
print("\nCombined DataFrame with Additional Information:")
print(df_combined.head())

# Visualization Section
sns.pairplot(df_cleaned[['age', 'english.grade', 'math.grade', 'sciences.grade', 'language.grade']])
plt.title('Pairplot of Grades and Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='portfolio_rating', data=df_cleaned, palette='coolwarm', estimator='mean')
plt.title('Average Portfolio Rating by Gender')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='nationality', y='age', data=df_cleaned, palette='Set2')
plt.xticks(rotation=90)
plt.title('Boxplot of Age by Nationality')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='age', y='english.grade', data=df_cleaned, marker='o', label='English')
sns.lineplot(x='age', y='math.grade', data=df_cleaned, marker='o', label='Math')
sns.lineplot(x='age', y='sciences.grade', data=df_cleaned, marker='o', label='Sciences')
plt.title('Grade Trends by Age')
plt.xlabel('Age')
plt.ylabel('Grade (Scaled)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
gender_counts = df_cleaned['gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
plt.title('Gender Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='gender', y='english.grade', data=df_cleaned, palette='muted')
plt.title('Distribution of English Grades by Gender')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[['age', 'english.grade', 'math.grade', 'sciences.grade', 'language.grade', 
                        'portfolio_rating', 'coverletter_rating', 'refletter_rating']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='nationality', data=df_cleaned, palette='magma')
plt.title('Count of Nationalities')
plt.xticks(rotation=90)
plt.show()

# Save the cleaned and combined data to a new CSV file
cleaned_file_path = 'cleaned_combined_file.csv'
df_combined.to_csv(cleaned_file_path, index=False)

# Summary of cleaning operations
print("\nSummary of Data Cleaning:")
print(f"Number of rows before cleaning: {len(df)}")
print(f"Number of rows after cleaning: {len(df_cleaned)}")
print(f"Number of rows after combining: {len(df_combined)}")
