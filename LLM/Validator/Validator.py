import os
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

import torch


hf_llama_models = { 'tokenizer': LlamaTokenizer, 'model': LlamaForCausalLM}
hf_auto_models = { 'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM}

backends = {'hf_llama': hf_llama_models, 'hf_auto': hf_auto_models}

class Validator(object):
    """
    LLM Validator: validate plans using LLM
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, arg, is_log_example = False, temperature = 0, device='cuda:0',max_len=512, max_new_tokens=100, backend_name='hf_auto', use_same_llm = False, llm=None):

        self.arg = arg
        self.model = arg.model
        self.temperature = temperature
        self.device=device
        self.messages = None
        self.log_dir = arg.logdir
        self.log_file_path = self.log_dir + "/validator_log.txt" 
        self.backend_name = backend_name
        self.max_new_tokens = max_new_tokens
        self.backend = backends[backend_name]
        self.tokenizer = self.backend['tokenizer'].from_pretrained(self.model)
        if not use_same_llm:
            self.llm = self.backend['model'].from_pretrained(
                self.model, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto"
            )
        elif llm is not None:
            self.llm = llm
        else:
            raise ValueError("llm is None")
        self.max_len = max_len
        self.is_log_example = is_log_example
        #self.pipepline = transformers.pipelines( )
        
        # root for prompt examples
        if self.arg.domain == 'blocksworld':
            self.max_examples = 6
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            if arg.method == "LLM_no_trans" or arg.method == "LLM_no_trans_self_feedback":
                self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_no_trans_examples")
            else:
                self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_examples")
        elif self.arg.domain == 'ballmoving':
            self.max_examples = 5
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ballmoving_examples")
        elif self.arg.domain == 'cooking':
            self.max_examples = 4
            self.num_valid_example = min(arg.num_valid_example, self.max_examples)
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
        for i in range(self.num_valid_example):

            file_path = self.prompt_example_root + "/example"+str(i)+".txt"
            with open(file_path, 'r') as f:
                contents = f.read().split('Answer:', 1)
                question = contents[0]
                answer = 'Answer:' + contents[1]

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
        # add new question to message list
        question_message = {"role": "user", "content": content}
        if is_append == False:
            question = self.messages.copy()
        else:
            question = self.messages
        question.append(question_message)
        self.write_content(content= content, is_append=True)

    

        pre_tokens =  self.tokenizer.apply_chat_template(question, tokenize=False,add_generation_prompt=True, )
        inputs = self.tokenizer(pre_tokens, return_tensors="pt", padding=False, truncation=True, max_length=2500).to(self.device)
        del pre_tokens
        outputs = self.llm.generate(
        inputs.input_ids,
        top_k=256,
        max_new_tokens=self.max_new_tokens,
        output_logits=False, 
        output_hidden_states=False,
        output_attentions=False,
        return_dict_in_generate=True,
        pad_token_id=self.tokenizer.eos_token_id,
        attention_mask = inputs.attention_mask,
        )
        len_question_tokens = len(inputs[0])
        generated_tokens = outputs.sequences[0][len_question_tokens:]
        generated_text = self.tokenizer.decode(generated_tokens,skip_special_tokens=True)

        response_content = generated_text
        self.write_content(content= response_content, is_append=True)
      


        return response_content


        