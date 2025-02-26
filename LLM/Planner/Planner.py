import os
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch


hf_llama_models = { 'tokenizer': LlamaTokenizer, 'model': LlamaForCausalLM}
hf_auto_models = { 'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM}

backends = {'hf_llama': hf_llama_models, 'hf_auto': hf_auto_models}

class Planner(object):
    """
    LLM Planner: generate plans
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, arg, model, is_log_example = False, temperature = 0.0001, device = None,max_len = 512, max_new_tokens = 100,
                 backend_name = 'hf_auto', use_same_llm = False, llm=None, output_hidden_states = False, output_attentions = False, 
                 output_logits = False, device_map = "auto", quant_8bit=False, debug=False):

        self.arg = arg
        self.model =    model
        self.temperature = temperature
        self.device = device
        self.messages = None
        self.log_dir = arg.logdir
        self.log_file_path = self.log_dir + "/planner_log.txt" 
        self.backend_name = backend_name
        self.max_new_tokens = max_new_tokens
        self.backend = backends[backend_name]
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.output_logits = output_logits
        #device ovverides device_map
        self.device = device
        self.device_map = device_map
        self.temperature = temperature
        self.tokenizer = self.backend['tokenizer'].from_pretrained(self.model)
        
        if not use_same_llm:
            if quant_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.llm = self.backend['model'].from_pretrained(
                    self.model, quantization_config=quantization_config , device_map =self.device_map,
                )
            else:
                self.llm = self.backend['model'].from_pretrained(
                    self.model, torch_dtype = torch.float16, device_map = self.device_map,
                )
        elif llm is not None:
            self.llm = llm
        else:
            raise ValueError("llm is None")
        self.max_len = max_len
        self.is_log_example = is_log_example
        
        # root for prompt examples
        if self.arg.domain == 'blocksworld':
            self.max_examples = 5
        else:
            self.max_examples = 3
        self.num_plan_example = min(arg.num_plan_example, self.max_examples)
        if arg.method == "LLM_no_trans" or arg.method == "LLM_no_trans_self_feedback":
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{arg.domain}_no_trans_examples")
        elif arg.method == "LLM_no_trans_simp":
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{arg.domain}_no_trans_simp_examples")
        else:
            self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{arg.domain}_examples")
       
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
        for i in range(self.num_plan_example):

            file_path = self.prompt_example_root + "/example"+str(i)+".txt"
            with open(file_path, 'r') as f:
                contents = f.read().split('Action Sequence', 1)
                question = contents[0]
                answer = 'Action Sequence' + contents[1]

                question_message = {"role": "system", "name":"example_user", "content": question}
                self.messages.append(question_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= question, is_append=True)
             
                    #
                answer_message = {"role": "system", "name":"example_assistant", "content": answer}
                self.messages.append(answer_message)  
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= answer, is_append=True)
        
        if "mistral" or "gemma" or "codellama" in self.model:
            # for mistral model, only one system call is allowed, those all previous messages must be merged into one
            total_sys_content = self.messages[0]["content"] + '\n'
            for message in self.messages[1:]:
            
                if message["name"] == "example_user":
                    total_sys_content += "Example of prompt:" + message["content"] + "\n"
                elif message["name"] == "example_assistant":
                    total_sys_content += "Example of expected answer:" + message["content"] + "\n"
            if "mistral" in self.model:
                self.messages = [{"role": "system", "content": total_sys_content}]
            else:
                self.messages = [{"role": "user", "content": total_sys_content}]
        

                        

    # Query question message
    def query(self, content, is_append = False, temperature = 0.1, debug=False):
        # add new question to message list
        if debug:
            print("content: ", content)
        # if only one message, then the message is the openning message
        if len(self.messages) == 1 and ("gemma" in self.model or "codellama" in self.model):
            openning_content = self.messages[0]["content"]
            question_message = {"role": "user", "content": openning_content + "\n" + content}
            self.messages = [question_message]
            if is_append == False:
                question = self.messages.copy()
            else:
                question = self.messages
                

        else:
            question_message = {"role": "user", "content": content}
            if is_append == False:
                question = self.messages.copy()
            else:
                question = self.messages
                question.append(question_message)
        
        self.write_content(content= content, is_append=True)
        pre_tokens =  self.tokenizer.apply_chat_template(question, tokenize=False,add_generation_prompt=True, )
        inputs = self.tokenizer(pre_tokens, return_tensors="pt", padding=False, truncation=True, max_length=self.max_len).to(self.device)
        del pre_tokens
        if "deepseek" in self.model:
            self.temperature = max(0.5, self.temperature)
        if self.temperature is not None and self.temperature>0:
            do_sample = True
        else:
            do_sample = False
        if "gemma" in self.model:
            self.llm = self.llm.bfloat16()
            
        outputs = self.llm.generate(
        inputs.input_ids,
        top_k = 128,
        max_new_tokens = self.max_new_tokens,
        output_logits = self.output_logits, 
        output_hidden_states = self.output_hidden_states,
        output_attentions = self.output_attentions,
        return_dict_in_generate = True,
        pad_token_id = self.tokenizer.eos_token_id,
        attention_mask = inputs.attention_mask,
        do_sample = do_sample, 
        temperature = self.temperature,)
        len_question_tokens = len(inputs[0])
        generated_tokens = outputs.sequences[0][len_question_tokens:]
        decoded_tokens = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
    
        # patch decoded tokens together into one string
        generated_text = ''.join(decoded_tokens)
    
    
        self.messages.append({"role": "assistant", "content": generated_text})
        self.write_content(content= generated_text, is_append=True)
        
        response= {'content':generated_text,'response_tokens':generated_tokens, "decoded_tokens":decoded_tokens, 
                   "input_tokens":inputs.input_ids[0]}
        if self.output_hidden_states:
            response['hidden_states'] = outputs.hidden_states
        if self.output_attentions:
            response['attentions'] = outputs.attentions
        if self.output_logits:
            response['logits'] = outputs.logits

        return response
