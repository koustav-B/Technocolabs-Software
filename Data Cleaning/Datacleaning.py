import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ask the user to input the path of the CSV file
file_path = input("Enter the path of your CSV file: ")

# Load the CSV file
df = pd.read_csv(file_path, encoding='utf-8')

# Display the first few rows of the original dataset
print("Original Data:")
print(df.head())

# Step 1: Handle missing values
# Option 1: Drop rows with any missing values
# df_cleaned = df.dropna()

# Option 2: Fill missing values for numerical columns with the mean, median, or a fixed value
df_cleaned = df.copy()
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].median())  # Fill age with median
df_cleaned['english.grade'] = df_cleaned['english.grade'].fillna(df_cleaned['english.grade'].mean())  # Fill with mean

# Option 3: Fill missing categorical values with mode
df_cleaned['city'] = df_cleaned['city'].fillna(df_cleaned['city'].mode()[0])

# Step 2: Remove duplicate rows (if any)
df_cleaned = df_cleaned.drop_duplicates()

# Step 3: Correct data types (example: converting latitude and longitude to floats)
df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')
df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')

# Step 4: Handle categorical data issues (e.g., inconsistent capitalization or spaces)
df_cleaned['gender'] = df_cleaned['gender'].str.strip().str.lower()  # Normalize gender entries
df_cleaned['nationality'] = df_cleaned['nationality'].str.strip().str.title()  # Capitalize nationality
df_cleaned['city'] = df_cleaned['city'].str.strip().str.title()  # Normalize city names

# Step 5: Remove outliers for age and grades (Optional: Example thresholds)
df_cleaned = df_cleaned[(df_cleaned['age'] >= 5) & (df_cleaned['age'] <= 100)]  # Remove unreasonable ages
df_cleaned = df_cleaned[(df_cleaned['english.grade'] >= 0) & (df_cleaned['english.grade'] <= 100)]
df_cleaned = df_cleaned[(df_cleaned['math.grade'] >= 0) & (df_cleaned['math.grade'] <= 100)]
df_cleaned = df_cleaned[(df_cleaned['sciences.grade'] >= 0) & (df_cleaned['sciences.grade'] <= 100)]
df_cleaned = df_cleaned[(df_cleaned['language.grade'] >= 0) & (df_cleaned['language.grade'] <= 100)]

# Step 6: Rename columns for consistency (optional)
df_cleaned = df_cleaned.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

# Step 7: Parse dates if there are any (Example: converting a 'date_of_birth' column)
# df_cleaned['date_of_birth'] = pd.to_datetime(df_cleaned['date_of_birth'], errors='coerce')

# Step 8: Handle inconsistent character encodings (ensure utf-8 encoding, or handle specific issues)
# (This step was already handled by reading the file with 'utf-8' encoding)

# Step 9: Scaling and normalization
# Example: Scaling numerical grades between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler()
df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']] = scaler.fit_transform(
    df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']]
)

# Example: Standardizing columns (mean = 0, std = 1) using StandardScaler
# scaler = StandardScaler()
# df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']] = scaler.fit_transform(
#     df_cleaned[['english.grade', 'math.grade', 'sciences.grade', 'language.grade']]
# )

# Step 10: Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_file.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

# Display the cleaned data
print("\nCleaned Data:")
print(df_cleaned.head())

# Summary of cleaning operations
print("\nSummary of Data Cleaning:")
print(f"Number of rows before cleaning: {len(df)}")
print(f"Number of rows after cleaning: {len(df_cleaned)}")
