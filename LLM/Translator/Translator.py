import os
from luna.utils.llama import LLaMATokenizer, LLaMAForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch

luna_models= {  'tokenizer': LLaMATokenizer, 'model': LLaMAForCausalLM}
hf_llama_models = { 'tokenizer': LlamaTokenizer, 'model': LlamaForCausalLM}
hf_auto_models = { 'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM}

backends = {'luna': luna_models, 'hf_llama': hf_llama_models, 'hf_auto': hf_auto_models}

class Translator(object):
    """
    LLM Translator: it translates description to PDDL domain file and problem file
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, arg, model, is_log_example = False, temperature = 0, device = None,max_len = 512,
                 backend_name = 'hf_auto', max_new_tokens = 100, output_hidden_states = False, 
                 output_attentions = False, output_logits = False, device_map="auto"):
        
        self.arg = arg
        self.model = model
        self.temperature = temperature
        self.messages = None
        self.log_dir = arg.logdir
        self.log_file_path = self.log_dir + "/translator_log.txt"
        self.is_log_example = is_log_example
        self.device= device
        self.max_len = max_len
        self.output_hidden_states = output_hidden_states
        self.output_attetions = output_attentions
        self.output_logits = output_logits
        #device ovverides device_map
        if device is not None:
            self.device_map = device
            self.device = device
        else:
            self.device_map = device_map
        self.backend = backends[backend_name]
        print(f"self.model: {self.model}")
        self.tokenizer = self.backend['tokenizer'].from_pretrained(self.model)
        self.llm = self.backend['model'].from_pretrained(self.model, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=self.device_map,
                                                          output_hidden_states=output_hidden_states)
        self.max_new_tokens = max_new_tokens
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

    

        pre_tokens =  self.tokenizer.apply_chat_template(question, tokenize=False,add_generation_prompt=True, )
        inputs = self.tokenizer(pre_tokens, return_tensors="pt", padding=False, truncation=True, max_length=2500).to(self.device)
        del pre_tokens
        outputs = self.llm.generate(
        inputs.input_ids,
        top_k = 256,
        max_new_tokens = self.max_new_tokens,
        output_logits = self.output_logits, 
        output_hidden_states = self.output_hidden_states,
        output_attentions = self.output_attentions,
        return_dict_in_generate = True,
        pad_token_id = self.tokenizer.eos_token_id,
        attention_mask = inputs.attention_mask,
        
        
        )
        len_question_tokens = len(inputs[0])
        generated_tokens = outputs.sequences[0][len_question_tokens:]
        generated_text = self.tokenizer.decode(generated_tokens,skip_special_tokens=True)
        self.write_content(content= generated_text, is_append=True)
        response= {'content':generated_text,'repsonse_tokens':generated_tokens}
        if self.output_hidden_states:
            response['hidden_states'] = outputs.hidden_states
        if self.output_attentions:
            response['attentions'] = outputs.attentions

        return response
        