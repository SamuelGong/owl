import os
import json
import time
import logging
import threading
import traceback

from datasets import load_dataset

from examples.run_ark import construct_society
from owl.utils import run_society


class GAIALoader:
    def __init__(self, level):
        # level: level1, level2, level3, all
        self.dataset = load_dataset(
            path="gaia-benchmark/GAIA",
            name=f"2023_{level}",
            trust_remote_code=True
        )

    def task2query(self, task, output_file):
        query = 'Your task is: {}'.format(task['Question'])
        if task['file_name']:
            query = query + (f'\n{task["file_path"]} is the absolute file path you need to use.')

        output_file = os.path.abspath(output_file)
        query += (f'\nWrite down your answer to file {output_file} with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. '
                  f'YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. '
                  f"If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
                  f"If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
                  f"If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.")

        return query


def set_log(log_path, log_level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # otherwise, run_gaia will produce messy logs

    logging.basicConfig(
        filename=log_path,
        level=log_level,
        format='[%(levelname)s][%(asctime)s.%(msecs)03d][%(process)d]'
               '[%(filename)s:%(lineno)d]: %(message)s',
        datefmt='(%Y-%m-%d) %H:%M:%S'
    )
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)


def blocking_run(query, result_holder):
    society = construct_society(query)
    answer, chat_history, token_count = run_society(society)
    result_holder["answer"] = answer
    result_holder["chat_history"] = chat_history
    result_holder["token_count"] = token_count


def call_agent(query, log_path):
    set_log(log_path)
    logging.info(f"Starting serving the query: {query}")
    # society = construct_society(query)
    # answer, chat_history, token_count = run_society(society)

    # detour to avoid errors:
    # playwright._impl._errors.Error: It looks like you are using Playwright Sync API inside the asyncio loop.
    # Please use the Async API instead.
    result = {}
    t = threading.Thread(target=blocking_run, args=(query, result))
    t.start()
    t.join()


def read_final_answer(output_path):
    with open(output_path, 'r') as f:
        final_answer = f.read()
    if "FINAL ANSWER: " in final_answer:
        if final_answer.startswith("['"):  # tailored for owl (may it is with doubao)
            final_answer = final_answer[2:-2]
        final_answer = final_answer.split("FINAL ANSWER: ")[1]
    else:
        final_answer = None
    return final_answer


def main():
    # set_type = "test"  # or "validation"
    set_type = "validation"
    result_file = f"gaia_{set_type}.jsonl"
    retry_limit = 3

    # Load processed task_ids if result_file already exists.
    processed_tasks = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_tasks.add(record["task_id"])
                except json.JSONDecodeError:
                    continue  # Skip any malformed lines

    # Process tasks and update result file incrementally.
    with open(result_file, 'a') as out_file:
        for level in ["level1", "level2", "level3"]:
            print(f"Processing {level}")
            gaia = GAIALoader(level)
            task_list = gaia.dataset[set_type]

            task_list_len = len(task_list)
            for task_idx, task in enumerate(task_list):
                task_id = task.get("task_id")

                if task_id in processed_tasks:
                    print(f"\t({task_idx + 1}/{task_list_len}) "
                          f"Skipping task {task_id} (already processed).")
                    continue
                print(f"\t({task_idx + 1}/{task_list_len}) "
                      f"Processing task {task_id}")
                log_dir = os.path.join('logs', f"{set_type}-{level}")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{task_id}.log")
                output_path = os.path.join(log_dir, f"{task_id}.txt")
                query = gaia.task2query(task, output_path)

                retry_time = 0
                final_answer = ""
                while retry_time < retry_limit:
                    retry_time += 1

                    start_time = time.perf_counter()
                    try:
                        call_agent(query, log_path)
                    except Exception as e:
                        print(f"Retrying task {task_id} for the {retry_time}th time due to error {e}")
                        print(traceback.format_exc())
                        continue

                    # comment out as now we are in relax mode
                    if not os.path.exists(output_path):
                        print(f"Retrying task {task_id} for the {retry_time}th time "
                              f"as no output file generated")
                        continue
                    else:
                        final_answer = read_final_answer(output_path)

                        if not final_answer:
                            print(f"Retrying task {task_id} for the {retry_time}th time "
                                  f"as final answer cannot be found in the output file")
                            continue
                    break

                end_time = time.perf_counter()
                duration = round(end_time - start_time, 3)

                record = {"task_id": task_id, "model_answer": final_answer}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

                processed_tasks.add(task_id)
                print(f"\tProcessed task {task_id} in {duration}s.")


if __name__ == '__main__':
    main()
