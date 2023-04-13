import pandas as pd
import random
import prompts

# load and prepare backstories/description given a data source. It is primarly used for the UQV100 but can be easily adapted to suite other data formats

class DataManager:

    def __init__(self, bs_path, var_path, data_src):
        self.data_src = data_src
        if self.data_src == "UQV":
            self.bs_id_col = "UQV100Id"
            self.bs_col= "Backstory"
            self.var_col= "variant"
        else: # used for other data sources
            self.bs_id_col = ""
            self.bs_col = ""
            self.var_col = ""
        self.bs_data = pd.read_csv(bs_path,sep="\t")
        self.var_data = pd.read_csv(var_path,sep="\t",names=[self.bs_id_col,self.var_col])


    def generate_prompts(self, examples, prompt_type):
        prompt_ls = []
        if prompt_type == "DESC_A":
            instruction = prompts.DESC_A
        elif prompt_type == "DESC_B":
            instruction = prompts.DESC_B
        elif prompt_type == "DESC_C":
            instruction = prompts.DESC_C
        else:
            instruction = prompts.DESC_def

        bs_ls = self.bs_data[self.bs_col].tolist()

        for bs in bs_ls:
            if len(examples) == 1:
                prompt = instruction + "\n\n" + examples[0] + "\n\n" + bs + " => "
            elif len(examples) == 3:
                prompt = instruction + "\n\n" + examples[0] + "\n\n" + examples[1] + examples[2] + "\n\n" + bs + " => "
            else:
                prompt = instruction + "\n\n" + bs
            prompt_ls.append(prompt)
        return prompt_ls

    def generate_random_example(self):
        bs_id_ls = self.bs_data[self.bs_id_col].tolist()
        # choose a random example (determenistic)
        random.seed(10)
        random_pos = random.randrange(1, len(bs_id_ls))

        random_id = bs_id_ls[random_pos]

        # get the bs
        bs = self.bs_data[self.bs_data[self.bs_id_col] == random_id][self.bs_col].iloc[0]
        # get variants
        variants = self.var_data[self.var_data[self.bs_id_col].apply(return_topic_id) == random_id][self.var_col].tolist()
        variants_st = "\n".join(variants)
        example = bs + " => \n" + variants_st
        return example


def return_topic_id(id):
    id_parts = str(id).split("|")
    return str(id_parts[2])
