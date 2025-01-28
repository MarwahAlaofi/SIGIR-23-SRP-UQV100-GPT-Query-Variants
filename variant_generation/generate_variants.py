import pandas as pd
import sys
import openai
from DataManager import *
import os
from dotenv import load_dotenv
from tqdm import tqdm

def main():
    # Check if the number of arguments is correct
    if len(sys.argv) != 2:
        print("Usage: python generate_variants.py temp")
        sys.exit(1)

    # Read the command line parameters
    temp = float(sys.argv[1])

    load_dotenv("api.env")
    api_key = os.environ.get('OPENAI_API_KEY')
    # load bs and variants using the DataManager class
    dm = DataManager("data/uqv100-backstories.tsv", "data/uqv100-systemInputRun-uniqueOnly-spelledNormQueries.tsv",
                     data_src="UQV")

    bs = dm.bs_data["Backstory"].tolist()
    # generate a random example to be added to the prompt
    example = dm.generate_random_example()
    example_emp = [example]
    # generate the inputs to the model
    prompt_ls = dm.generate_prompts(example_emp,"DESC_A")

    print("# of prompts to be submitted: {}".format(len(prompt_ls)))
    # for i in prompt_ls:
    #     print(i)
    print(" ========================= ")
    print("Do you want to continue(y/n)?")
    ans = input()
    if ans == "n":
        exit()

    generated_variants = []
    uqv_ids = []
    bs_ls = []

    # Set up the OpenAI Client
    client = openai.OpenAI(api_key=api_key)
    MODEL = "gpt-3.5-turbo"
    MAX_TOK = 3000

    try:
        for i in tqdm(range(0,len(prompt_ls))):
            # print(example)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[ {
                        "role" : "user",
                        "content": prompt_ls[i],
                           }
                          # could also have a system prompt if desired
                         ],
                temperature=temp,
                max_tokens=MAX_TOK
            )
            response = response.choices[0].message.content.strip()
            for ln in response.split("\n"):
                generated_variants.append(ln)
                uqv_ids.append("UQV100." + str(i + 1).zfill(3))
                bs_ls.append(bs[i])
    except Exception as e:
        print("Error connecting to openai.. " + str(e))
    finally:
        generated_data = pd.DataFrame({"UQV100Id": uqv_ids, "backstory": bs_ls,"query": generated_variants})
        generated_data.to_csv(dm.data_src + "_GPT_variants_temp_" + str(temp))

if __name__ == "__main__":
    main()
