import random
import pandas as pd

def make_data(max_digits: int = 10, iters_per_comb: int = 10, outpath: str = "data/numbers.csv"):
    """
    For any task, we need to generate data for evaluation.
    We do this up to max_digit numbers. We want 10 examples
    of each of the length of the two numbers — so we generate
    10 * max_digits * max_digits examples.
    """
    data = []
    for i in range(1, max_digits + 1):
        for j in range(1, max_digits + 1):
            for _ in range(iters_per_comb):
                number1 = random.randint(10 ** (i - 1), 10 ** i)
                number2 = random.randint(10 ** (j - 1), 10 ** j)
                data.append({
                    "number1": number1,
                    "number2": number2,
                    "length1": i,
                    "length2": j
                })
    
    # Cast to pandas DataFrame
    df = pd.DataFrame(data, columns=["number1", "number2", "length1", "length2"])
    df.to_csv(outpath, index=False)
    return df

if __name__ == "__main__":
    make_data(iters_per_comb=100, outpath="data/numbers_100.csv")