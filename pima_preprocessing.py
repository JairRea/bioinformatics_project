import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class PimaDiabetesPreprocessor:
    
    def __init__(self, data_path='data/pima_diabetes.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                f"Please ensure the Pima Diabetes dataset is in the data/ folder."
            )
        
        df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df):
        print("\n=== Dataset Overview ===")
        print(df.head())
        print("\n=== Data Types ===")
        print(df.dtypes)
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        print("\n=== Target Distribution ===")
        print(df['Outcome'].value_counts())
        print("\n=== Statistical Summary ===")
        print(df.describe())
        
    def clean_data(self, df):
        # Remove duplicates
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - df.shape[0]
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle zeros in certain features (zeros are biologically implausible for these)
        # In Pima dataset, 0 often represents missing values for certain features
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for feature in zero_features:
            if feature in df.columns:
                zero_count = (df[feature] == 0).sum()
                if zero_count > 0:
                    print(f"Found {zero_count} zero values in {feature}, replacing with median")
                    median_value = df[df[feature] != 0][feature].median()
                    df.loc[df[feature] == 0, feature] = median_value
        
        # Handle missing values (if any)
        if df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # For numerical columns, fill with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def prepare_features(self, df, test_size=0.2, random_state=42):
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {len(self.feature_names)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names
    
    def preprocess(self, explore=True):
        # Load data
        df = self.load_data()
        
        # Explore (optional)
        if explore:
            self.explore_data(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Prepare features and split data
        return self.prepare_features(df)


def get_pima_preprocessed_data(data_path='data/pima_diabetes.csv', explore=False):
    preprocessor = PimaDiabetesPreprocessor(data_path)
    return preprocessor.preprocess(explore=explore)


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing Pima Diabetes Preprocessing Pipeline...")
    try:
        X_train, X_test, y_train, y_test, feature_names = get_pima_preprocessed_data(explore=True)
        print("\n=== Preprocessing Successful ===")
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Feature names: {feature_names}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
