import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_train_data(file_path='../data/train.csv'):
    """Load the train.csv dataset"""
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Display basic info about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the dataset"""
    logger.info("Cleaning data...")
    
    initial_count = len(df)
    
    # Remove rows with missing values
    df = df.dropna(subset=['Context', 'Response'])
    logger.info(f"Removed {initial_count - len(df)} missing values")
    
    # Remove rows with empty strings
    df = df[df['Context'].str.strip() != '']
    df = df[df['Response'].str.strip() != '']
    logger.info(f"Removed {initial_count - len(df)} empty rows")
    
    # Remove duplicates
    # initial_count = len(df)
    # df = df.drop_duplicates(subset=['Context', 'Response'])
    # logger.info(f"Removed {initial_count - len(df)} duplicate rows")
    
    # Add unique IDs
    df = df.reset_index(drop=True)
    df['id'] = [f"conv_{i:06d}" for i in range(len(df))]
    
    # Rename columns
    df = df.rename(columns={
        'Context': 'context',
        'Response': 'response'
    })
    
    logger.info(f"Final dataset: {len(df)} rows")
    return df


def save_to_sqlite(df, db_path='../data/processed.db'):
    """Store the processed data in SQLite"""
    logger.info(f"Saving data to SQLite database: {db_path}")
    
    try:
        # Create SQLAlchemy engine
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Save DataFrame to SQL table
        df.to_sql('db', engine, index=False, if_exists='replace')
        
        # Create indexes for better performance
        with engine.connect() as conn:
            # Create indexes on commonly searched columns
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_context ON db(context)'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_response ON db(response)'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_id ON db(id)'))
            conn.commit()
        
        # Verify the data was stored correctly
        verification_df = pd.read_sql('SELECT COUNT(*) as count FROM db', engine)
        logger.info(f"Successfully stored in SQLite database")
        
        return engine
        
    except Exception as e:
        logger.error(f"Error saving to SQLite: {e}")
        return None

def save_processed_csv(df, file_path='../data/processed.csv'):
    """Save processed data as CSV for backup"""
    logger.info(f"Saving processed data to CSV: {file_path}")
    df.to_csv(file_path, index=False)
    logger.info("CSV file saved successfully")

def run_pipeline():
    """Run the complete data pipeline"""
    logger.info("Starting data pipeline...")
    
    try:
        # Load data
        df = load_train_data()
        
        # Clean data
        df = clean_data(df)
        
        # Save to SQLite
        engine = save_to_sqlite(df)
        
        # Save CSV
        save_processed_csv(df)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return None
    
    logger.info("Pipeline completed successfully!")
    
    return df

if __name__ == "__main__":
    # Run the pipeline
    result = run_pipeline()
    
    if result is not None:
        print("\nPipeline completed successfully!")
    else:
        print("\nPipeline failed.")