import os
from luna.utils.llama import LLaMATokenizer, LLaMAForCausalLM
import torch

class Translator(object):
    """
    LLM Translator: it translates description to PDDL domain file and problem file
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, arg, is_log_example = False, temperature = 0,device='cuda:0',max_len=512):
        
        self.arg = arg
        self.model = arg.model
        self.temperature = temperature
        self.messages = None
        self.log_dir = arg.logdir
        self.log_file_path = self.log_dir + "/translator_log.txt"
        self.is_log_example = is_log_example
        self.device= device
        self.max_len = max_len
        self.tokenizer = LLaMATokenizer.from_pretrained(self.model)
        self.llm = LLaMAForCausalLM.from_pretrained(
            self.model, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=self.device
        )
        
        # root for prompt examples
        if self.arg.domain == 'blocksworld':
            self.max_examples = 3
            self.num_trans_example = min(arg.num_trans_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_examples")
        elif self.arg.domain == 'ballmoving':
            self.max_examples = 3
            self.num_trans_example = min(arg.num_trans_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ballmoving_examples")
        elif self.arg.domain == 'cooking':
            self.max_examples = 3
            self.num_trans_example = min(arg.num_trans_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cooking_examples")
        
        # initialize messages
        self.init_messages()

    # Write content to file 
    def write_content(self, content, is_append):

        if is_append == False:

            with open(self.log_file_path, "w") as f:
                f.write(content+"\n")

        else:

            with open(self.log_file_path, "a") as f:
                f.write(content+"\n")

    # Initialize messages include opening and few-shot examples
    def init_messages(self, is_reinitialize = False):

        # opening setup
        file_path = self.prompt_example_root + "/opening.txt"
        with open(file_path, 'r') as f:
            contents = f.read()
            opening_message =  {"role": "system", "content": contents}
            self.messages = [opening_message]
            # record content
            if self.is_log_example == True and is_reinitialize == False:
                self.write_content(content= contents, is_append=False)

        # load few-shot examples
        for i in range(self.num_trans_example):

            file_path = self.prompt_example_root + "/example"+str(i)+".txt"
            with open(file_path, 'r') as f:
                contents = f.read().split('\n', 1)
                question = contents[0]
                answer = contents[1]
                
                question_message = {"role": "system", "name":"example_user", "content": question}
                self.messages.append(question_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= question, is_append=True)

                answer_message = {"role": "system", "name":"example_assistant", "content": answer}
                self.messages.append(answer_message)   
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= answer, is_append=True)

    # Query question message
    def query(self, content, is_append = False):

        # add new question to message list
        question_message = {"role": "user", "content": content}
        if is_append == False:
            question = self.messages.copy()
        else:
            question = self.messages
        question.append(question_message)
        self.write_content(content= content, is_append=True)

        question_text = content
        # Translate description to PDDL domain and problem files
        question_tokens = self.tokenizer.encode(question_text, return_tensors="pt").to(
            self.device
        )
        generated_tokens = self.llm.generate(
        question_tokens,
        top_k=128,
        max_length=self.max_len,
        num_return_sequences=1,
        output_scores=True,
        output_hidden_states=True,
        output_attentions=True,
        return_dict_in_generate=True,
        )
        len_question_tokens = len(question_tokens[0])
        generated_tokens = generated_tokens.sequences[0][len_question_tokens:]
        generated_text = self.tokenizer.decode(generated_tokens,skip_special_tokens=True)

        response_content = generated_text
        self.write_content(content= response_content, is_append=True)
      

        return response_content
        