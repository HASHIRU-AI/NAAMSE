import linecache
import time
import os
import random
import pandas as pd
from src.cluster_engine.data_access import create_data_source
from src.cluster_engine.utilities import get_random_prompt

CHUNK_SIZE = 1000
PATH = ".\src\cluster_engine\jailbreak_corpus.jsonl"


def get_number_lines(filename):
    with open(filename, "rb") as f:
        num_lines = sum(1 for _ in f)
        return num_lines


def run_function_with_time(func):
    start = time.time()
    func()
    end = time.time()
    return end - start


def get_line_using_readlines(filename):
    n = get_number_lines(filename) - 1
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if 0 <= n < len(lines):
            return lines[n].strip()
        else:
            return None


def get_line_using_loop(filename):
    n = get_number_lines(filename) - 1
    with open(filename, "r", encoding="utf-8") as f:
        for current_line_number, line in enumerate(f):
            if current_line_number == n:
                return line.strip()


def get_line_using_islice(filename):
    n = get_number_lines(filename) - 1
    from itertools import islice
    with open(filename, "r", encoding="utf-8") as f:
        line = next(islice(f, n, n+1), None)
        return line.strip() if line else None


def get_line_using_linecache(filename):
    n = get_number_lines(filename) - 1
    line = linecache.getline(filename, n+1)
    return line.strip() if line else None


def get_random_line_using_reservoir_sampling(filename):
    selected_line = None
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if random.randrange(index) == 0:
                selected_line = line
    return selected_line.strip() if selected_line else None


def check_prompt_exists(filename):
    df = pd.read_json(filename, lines=True)
    # print(df.head())
    # get line count
    line_count = len(df)
    print(f"Line count: {line_count}")
    # data_source = create_data_source('jsonl')
    # return data_source.check_prompt_exists(filename)


line_functions = {
    # "readlines": get_line_using_readlines,
    # "loop": get_line_using_loop,
    # "islice": get_line_using_islice,
    # "linecache": get_line_using_linecache,
    # "reservoir_sampling": get_random_line_using_reservoir_sampling,
    # "random_prompt": get_random_prompt,
    "check_prompt_exists": check_prompt_exists,
    # "reservoir_sampling_lib": get_random_line_using_reservoir_sampling_lib
}

# run each function 10 times and average the time taken
for func_name, func in line_functions.items():
    total_time = 0.0
    runs = 10
    for _ in range(runs):
        total_time += run_function_with_time(lambda: func(PATH))
    average_time = total_time / runs
    print(
        f"Average time using {func_name}: {average_time:.6f} seconds over {runs} runs")
