Sure, here is the `reproducer.py` file that you can use to reproduce the issue:

```python
import pandas as pd

def print_stacktrace(e: Exception):
    import traceback
    import sys
    tb = traceback.extract_tb(e.__traceback__)
    print("Traceback (most recent call last):", file=sys.stderr)
    for frame in tb:
        line_number = frame.lineno
        code_context = frame.line.strip() if frame.line else "Unknown"
        print(f'  File "{frame.filename}"', file=sys.stderr)
        print(f"    {line_number}: {code_context}", file=sys.stderr)
    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

def main():
    try:
        df = pd.DataFrame({'a': range(5), 'b': range(5)})

        # let's insert some -1 values
        df.loc[1, 'a'] = -1
        df.loc[1, 'b'] = -1
        df.loc[3, 'a'] = -1
        df.loc[4, 'b'] = -1

        df1 = df[(df.a != -1) & (df.b != -1)]
        df2 = df[(df.a != -1) | (df.b != -1)]

        expected_and_df = pd.DataFrame({'a': [0, 2], 'b': [0, 2]})
        expected_or_df = pd.DataFrame({'a': [0, 2, -1, 4], 'b': [0, 2, 3, -1]})

        assert df1.equals(expected_and_df), "AND condition did not filter as expected"
        assert df2.equals(expected_or_df), "OR condition did not filter as expected"

    except AssertionError as e:
        print_stacktrace(e)
        raise

if __name__ == '__main__':
    main()
```

With this code:
- The `AssertionError` will be raised if the condition filtering using `&` or `|` does not work as expected.
- The `print_stacktrace` function will print detailed stack trace information for debugging purposes.
- If the filtering works as expected, the script will exit with code 0.