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
