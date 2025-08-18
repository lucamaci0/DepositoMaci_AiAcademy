import os
import string
from collections import Counter

def summary_input_file(file_name: str, n_most_common_words: int = 5):
    """
    Read the contents of a text file and prints summary statistics about it.

    Args:
        file_name: name of the file to be analyzed. File should be in the same directory as the Text_Analyzer.py file.
        n_most_common_words: amount of the most frequent words to print.
    Returns:
        None

    Raises:
        FileNotFoundError: If the path doesn't exist or isn't a file.
    """

    directory_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.expanduser(file_name)
    full_path = os.path.normpath(os.path.join(directory_path, file_path))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Path not found: {full_path}")
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Not a file: {full_path}")
    
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

        lines = text.splitlines()
        n_lines = len(lines)
        print(f"The input text contains {n_lines} lines")

        separate_punctuations = str.maketrans({c: " " for c in string.punctuation})
        words = " ".join(text.translate(separate_punctuations).split()).lower().split(" ")
        n_words = len(words)
        print(f"The input text contains {n_words} words (we're separating punctuation-joined-words and counting them separately)")
        
        words_counter = Counter(words)
        most_common_words = words_counter.most_common(n_most_common_words)
        print(f"The top {n_most_common_words} most common words are: {most_common_words}")


if __name__ == "__main__":
  
  input_path = "input.txt"

  try:
      print("Generating summary...")
      summary_input_file(input_path)
  except Exception as e:
      print(f"Error: {e}")