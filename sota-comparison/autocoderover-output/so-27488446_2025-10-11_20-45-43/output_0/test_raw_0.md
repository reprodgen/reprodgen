Sure, here is the `reproducer.py` script which will reproduce the issue and print a stack trace if the expected word frequencies do not match the computed frequencies.

```python
import traceback
import sys
from sklearn.feature_extraction.text import CountVectorizer

def print_stacktrace(e: Exception):
    tb = traceback.extract_tb(e.__traceback__)
    print("Traceback (most recent call last):", file=sys.stderr)
    for frame in tb:
        line_number = frame.lineno
        code_context = frame.line.strip() if frame.line else "Unknown"
        print(f'  File "{frame.filename}"', file=sys.stderr)
        print(f"    {line_number}: {code_context}", file=sys.stderr)
    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

def main():
    texts = ["dog cat fish", "dog cat cat", "fish bird", "bird"]
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(texts)
    
    # Sum up the counts of each vocabulary word
    word_counts = cv_fit.toarray().sum(axis=0)
    
    # Map the counts to words
    word_freq = {word: word_counts[idx] for word, idx in cv.vocabulary_.items()}
    
    # Expected word frequencies
    expected_word_freq = {"bird": 2, "cat": 3, "dog": 2, "fish": 2}
    
    try:
        assert word_freq == expected_word_freq, f"Expected {expected_word_freq} but got {word_freq}"
    except AssertionError as e:
        print_stacktrace(e)
        raise e

if __name__ == "__main__":
    main()
```

To use this script, simply save it as `reproducer.py` in the root directory of your project. When you run `python3 reproducer.py`, it will raise an `AssertionError` if the issue is present and print the associated stack trace using the provided `print_stacktrace` function. If the issue has been fixed, the script will exit with code 0 and no `AssertionError` will be raised.