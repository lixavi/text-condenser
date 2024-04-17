from utils.text_processing import preprocess_text
from utils.summarization import generate_summary

def main():
    # Read input text
    input_text_path = "data/input_text.txt"
    with open(input_text_path, "r", encoding="utf-8") as file:
        input_text = file.read()

    # Preprocess text
    preprocessed_text = preprocess_text(input_text)

    # Generate summary
    summary = generate_summary(preprocessed_text)

    # Save summary
    summary_output_path = "data/summaries/summary.txt"
    with open(summary_output_path, "w", encoding="utf-8") as file:
        file.write(summary)

    print("Summary generated and saved successfully!")

if __name__ == "__main__":
    main()

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess the loaded data.

    Parameters:
    - data (DataFrame): The loaded data as a pandas DataFrame.

    Returns:
    - DataFrame: The preprocessed data.
    """
    # Add preprocessing steps here, such as handling missing values, encoding categorical variables, etc.
    preprocessed_data = data.dropna()  # Example: Drop rows with missing values
    return preprocessed_data

