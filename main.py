import os
import argparse
import datetime
import numpy as np
import time
import re

from utils.utils import *
from LLM.Translator.Translator import Translator
from LLM.Planner.Planner import Planner
from LLM.Validator.Validator import Validator
from Blocks_Sim.Block_Sim import BlockSim
from BallMoving_Sim.BallMoving_Sim import BallMovingSim
from Cooking_Sim.Cooking_Sim import CookingSim

DOMAINS = ["blocksworld", "ballmoving", "cooking"]
METHODS = ["LLM_trans_self_feedback", "LLM_trans_no_feedback", "LLM_trans_exact_feedback", "LLM_no_trans", "LLM_no_trans_simp", "LLM_no_trans_self_feedback"]
MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct","mistralai/Mistral-Nemo-Instruct-2407",""]
MODEL_NAMES = ["llama_8B", "llama_3B", "mistral_nemo", "gemma_9B", "gemma_27B", "gemma_7B", "qwen_14B", "qwen_7B"]
model_dict= {"llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
             "meta_8B": "meta-llama/Llama-3.1-8B-Instruct",
              "llama_3B": "meta-llama/Llama-3.2-3B-Instruct", 
              "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
              "gemma_9B": "google/gemma-2-9b-it",
              "gemma_27B": "google/gemma-2-27b-it",
              "gemma_7B": "google/gemma-7b-it",
              "qwen_14B": "Qwen/Qwen2.5-14B-Instruct",
              "qwen_7B": "Qwen/Qwen2.5-7B-Instruct"}
# LLM planning without PDDL translator
def test_LLM_no_trans(test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset,
                                  test_log_file_path, gpt_api_wait_time, print_states = False, simple = False,):
    nb_tot_attempt = 0
    nb_total_errors = 0
    nb_success = 0
    for i in range(num_test):
    # test loop
        

        # wait for every loop (gpt api has rpm limit)
        #time.sleep(gpt_api_wait_time) not needed with open-source models

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)
        
        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        planning_problem = description + '\n'

        # refine loop
        for j in range(max_num_refine + 1):

            #time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            nb_tot_attempt+=1
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            # increase temperature if more attempts
            if i == 0: 
                temperature = 0.0001
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner['content']

            # simulate actions
            print(f"Total nb of attempts: {nb_tot_attempt}")
            print(f"Cumulative error rate: {(nb_total_errors/ nb_tot_attempt)*100} %")
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # validation using external simulation
            is_satisfied, is_error, error_message, error_action, error_type, states, actions = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)
            if print_states:
                for i in range(len(states)):
                    print(f"State {i}: {states[i]}")
            # if error is found 
            if is_error == True:
                nb_total_errors+=1
                if is_satisfied == False:

                    # some action is wrong
                    if error_action != None:

                        error_description = "Action " + error_action + " is wrong. Error info: " + error_message
                        planning_problem = error_description + "Please find a new plan. "

                    # no correct action is returned by gpt response
                    else:

                        error_description = error_message
                        planning_problem = error_description + "Please find a new plan. "

                # if goal is satisfied before action sequence finishes
                else:

                    error_description = error_message + "Please ignore actions after action " + error_action
                    planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no action returned
            elif is_error == None:

                error_description = "Please find a new plan. "
                planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no error found
            else:
                nb_success+=1
                # exit refine loop
                break
        print(f"Success rate: {(nb_success/num_test)*100} %")
        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")


# LLM planning without PDDL translator and using self feedback
def test_LLM_no_trans_self_feedback(domain, test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time, print_states = False):
    nb_tot_attempt = 0
    nb_total_errors = 0
    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        #time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        planning_problem = description + '\n'
        # print(response_translator)

        # refine loop
        for j in range(max_num_refine + 1):

            #time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            temperature = min(max_refine_temperature, 0.1*j) 
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # self-evaluate actions
            action_description = extract_action_description(action_sequence, domain)

            if domain == 'blocksworld':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description
            elif domain == 'ballmoving':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description
            elif domain == 'cooking':
                validate_question = "Question:\n" + planning_problem + "Examined action sequence:\n" + action_description

            print(validate_question)
            with open(test_log_file_path, "a") as f:
                f.write(validate_question +"\n")

            #time.sleep(gpt_api_wait_time)

            response_validator = LLM_Validator.query(validate_question, is_append=False)
            response_validator_content = response_validator
            with open(test_log_file_path, "a") as f:
                f.write(response_validator_content +"\n")

            # Answer
            valid_result = response_validator_content.split('Final answer:', 1)
            if len(valid_result) == 1: # no action returned
                break
                valid_result = 'No'
            else:
                valid_result = valid_result[1]

            if 'Yes' in valid_result:

                print("Self-evaluation suggests a solution.")
                with open(test_log_file_path, "a") as f:
                    f.write("Self-evaluation suggests a solution.\n")
                # exit self-refine loop
                break

            elif 'No' in valid_result:

                print("Self-evaluation suggests a failure.")
                error_description = "Goal is not satisfied." #Error analysis:" + summary_content 
                if domain == 'blocksworld':
                    planning_problem = error_description + " Please find a new plan by considering the goals from bottom to top. "
                elif domain == 'ballmoving':
                    planning_problem = error_description + " Please find a new plan by considering the locations of balls. "
                elif domain == 'cooking':
                    planning_problem = error_description + " Please find a new plan by considering the ingredients needed in each pot. "
                print(planning_problem)

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            else:

                print("Unknown results:", valid_result)

        # check actual results
        print("Actual analysis:")
        with open(test_log_file_path, "a") as f:
            f.write("Actual analysis:\n")
        is_satisfied, is_error, error_message, error_action, error_type, states, actions = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")

# LLM planning with PDDL translator and using exact feedback from external validator
def test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time, print_states = False):


    # test loop
    nb_total_errors = 0
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        #time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(f"Test case {i + num_prompt_examples_dataset}/{num_test}:")
        print(f"Desc: {description}")

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        response_translator = LLM_Translator.query(description, is_append = False)
        
        planning_problem = response_translator["content"]
        
        print(f"LLM-Translation: \n {planning_problem}")

        # refine loop
        for j in range(max_num_refine + 1):

            #time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            # increase temperature if more attempts
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            actions = response["content"]
            

            # LLM planner
            action_sequence = response["content"]

            #print(f"Response: \n {action_sequence}") 
            # simulate actions
            print("Attempt", j)
            print(f"Action sequence: \n {action_sequence}")
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # validation using external simulation
            is_satisfied, is_error, error_message, error_action, error_type, states, actions = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

            # if error is found
            if is_error == True:
                nb_total_errors+=1
                if is_satisfied == False:

                    # some action is wrong
                    if error_action != None:

                        error_description = "Action " + error_action + " is wrong. Error info: " + error_message
                        planning_problem = error_description + "Please find a new plan. "

                    # no correct action is returned by gpt response
                    else:

                        error_description = error_message
                        planning_problem = error_description + "Please find a new plan. "

                # if goal is satisfied before action sequence finishes
                else:

                    error_description = error_message + "Please ignore actions after action " + error_action
                    planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no action returned
            elif is_error == None:

                error_description = "Please find a new plan. "
                planning_problem = error_description

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            # if no error found
            else:

                # exit refine loop
                break

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")

# LLM planning with PDDL translator and using self feedback
def test_LLM_trans_self_feedback(domain, test_initial_state, test_goal_state, num_test, max_num_refine,
                                  max_refine_temperature, num_prompt_examples_dataset, 
                                  test_log_file_path, gpt_api_wait_time, print_states = False):


    # test loop
    for i in range(num_test):

        # wait for every loop (gpt api has rpm limit)
        #time.sleep(gpt_api_wait_time)

        initial_state = test_initial_state[i + num_prompt_examples_dataset, :]
        goal_state = test_goal_state[i + num_prompt_examples_dataset, :]

        # generate description
        description = scenario_simulator.generate_scene_description(initial_state= initial_state,
                                                        goal_state = goal_state, 
                                                        constraint = None)

        print(i + num_prompt_examples_dataset, description)

        with open(test_log_file_path, "a") as f:
            f.write("Test case index: " + str(i+num_prompt_examples_dataset) + "\n")
            f.write(description +"\n")

        # LLM translator
        response_translator = LLM_Translator.query(description, is_append = False)
        planning_problem = response_translator["choices"][0]["message"]["content"]
        # print(response_translator)

        # initial and goal state in pddl
        pddl_init_state, pddl_goal_state = extract_state_pddl(planning_problem, domain)

        # refine loop
        for j in range(max_num_refine + 1):

            #time.sleep(gpt_api_wait_time)

            # (re-)initialize block sim
            scenario_simulator.initialize_state(initial_state= initial_state, goal_state = goal_state, constraint = None)

            with open(test_log_file_path, "a") as f:
                f.write("Attempt: " + str(j) + "\n")

            # LLM planner
            if i == 0: 
                temperature = 0
            else:
                temperature = min(max_refine_temperature, 0.1*i)
            response_planner = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)
            # print(response_planner)

            # LLM planner
            action_sequence = response_planner["choices"][0]["message"]["content"]

            # simulate actions
            print("Attempt", j)
            print(action_sequence)
            with open(test_log_file_path, "a") as f:
                f.write(action_sequence +"\n")
                f.write("Analysis: " +"\n")

            # self-evaluate actions
            action_description = extract_action_description(action_sequence, domain)

            if domain == 'blocksworld':
                validate_question = "Question:\nBlock initial state:\n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description
            elif domain == 'ballmoving':
                validate_question = "Question:\nRobot and ball initial state: \n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description
            elif domain == 'cooking':
                validate_question = "Question:\nInitial state: \n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description

            print(validate_question)
            with open(test_log_file_path, "a") as f:
                f.write(validate_question +"\n")

            #time.sleep(gpt_api_wait_time)

            response_validator = LLM_Validator.query(validate_question, is_append=False)
            response_validator_content = response_validator["choices"][0]["message"]["content"]
            with open(test_log_file_path, "a") as f:
                f.write(response_validator_content +"\n")

            # Answer
            valid_result = response_validator_content.split('Final answer:', 1)
            if len(valid_result) == 1: # no action returned
                break
                valid_result = 'No'
            else:
                valid_result = valid_result[1]

            if 'Yes' in valid_result:

                print("Self-evaluation suggests a solution.")
                with open(test_log_file_path, "a") as f:
                    f.write("Self-evaluation suggests a solution.\n")
                # exit self-refine loop
                break

            elif 'No' in valid_result:

                print("Self-evaluation suggests a failure.")
                error_description = "Goal is not satisfied." #Error analysis:" + summary_content 
                if domain == 'blocksworld':
                    planning_problem = error_description + " Please find a new plan by considering the goals from bottom to top. "
                elif domain == 'ballmoving':
                    planning_problem = error_description + " Please find a new plan by considering the locations of balls. "
                elif domain == 'cooking':
                    planning_problem = error_description + " Please find a new plan by considering the ingredients needed in each pot. "
                print(planning_problem)

                with open(test_log_file_path, "a") as f:
                    f.write(planning_problem +"\n")

            else:

                print("Unknown results:", valid_result)

        # check actual results
        print("Actual analysis:")
        with open(test_log_file_path, "a") as f:
            f.write("Actual analysis:\n")
        is_satisfied, is_error, error_message, error_action, error_type, states, actions = scenario_simulator.simulate_actions(action_sequence, test_log_file_path)

        # reinitialize planner for next case
        LLM_Planner.init_messages(is_reinitialize = True)

        with open(test_log_file_path, "a") as f:
            f.write("End of test case " + str(i+num_prompt_examples_dataset) + "\n\n\n")


if __name__=="__main__":

  
    parser = argparse.ArgumentParser(description="LLM-Task-Planner")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="blocksworld")
    parser.add_argument('--method', type=str, choices=METHODS, default="LLM_trans_exact_feedback")
    parser.add_argument('--model', type=str, choices=MODELS, default="")
    parser.add_argument('--model_name', type=str, choices=MODEL_NAMES, default="meta_8B")
    parser.add_argument('--max_len', type=int, default=8000)
    parser.add_argument('--temperature', type=float, default=0.0001)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--backend', type=str, default="hf_auto")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--num_objects', type=int, choices=[3,4], default=3)
    parser.add_argument('--num_trans_example', type=int, choices=[1,2,3], default=3)
    parser.add_argument('--num_plan_example', type=int, choices=[3,4,5], default=4)
    parser.add_argument('--num_valid_example', type=int, choices=[4,5,6], default=6)
    parser.add_argument('--use_same_llm', action='store_true',default=True)
    parser.add_argument('--print_states', action='store_true',default=False)
    parser.add_argument('--debug', action='store_true',default=False)
    parser.add_argument('--num_test', type=int, default = 10)
    parser.add_argument('--gpt_api_wait_time', type=float, default=0.01)


    args = parser.parse_args()

    if args.model == "":
        args.model = model_dict[args.model_name]
    # initialize log dir
    if args.logdir == None:
        args.logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_log/")
        args.logdir = args.logdir + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
    if len(args.device) == 1:
        if args.device in ['0', '1', '2', '3']:
            args.device = "cuda:" + args.device
        else:
            args.device = "cuda:0"
    # Initialize translator
    LLM_Translator = Translator(args, model=args.model, is_log_example=True, max_len=args.max_len, backend_name=args.backend, max_new_tokens=args.max_new_tokens,
                                device = args.device)
    
    
    # Initialize planner
    if args.use_same_llm:
        LLM_Planner = Planner(args, model=args.model, is_log_example=True, llm=LLM_Translator.llm, use_same_llm=True, max_len=args.max_len, backend_name=args.backend, max_new_tokens=args.max_new_tokens,
                              device = args.device, debug = args.debug)
    else:
        LLM_Planner = Planner(args, model=args.model,  is_log_example=True, max_len=args.max_len, backend_name=args.backend, max_new_tokens=args.max_new_tokens,
                              device = args.device, debug = args.debug)

    # Initialize validator 
    if args.method == "LLM_trans_self_feedback" or args.method == "LLM_no_trans_self_feedback":
        if args.use_same_llm:
            LLM_Validator = Validator(args, is_log_example=True,llm = LLM_Translator.llm, use_same_llm=True, max_len=args.max_len, backend_name=args.backend, max_new_tokens=args.max_new_tokens)
        else:
            LLM_Validator = Validator(args, is_log_example=True, max_len=args.max_len, backend_name=args.backend, max_new_tokens=args.max_new_tokens)

    # Initialize block simulator
    if args.domain == 'blocksworld':
        scenario_simulator = BlockSim()

    elif args.domain == 'ballmoving':
        scenario_simulator = BallMovingSim()

    elif args.domain == 'cooking':
        scenario_simulator = CookingSim()


    ###############################################
    # load test scenarios
    test_initial_state, test_goal_state = load_test_scenarios(args)

    # run test
    num_test = args.num_test
    num_prompt_examples_dataset = 3 # the first n examples are in the prompt example, so skip them
    max_num_refine = 10  # max number of refinement, if it is 0 -> no feedback
    gpt_api_wait_time = args.gpt_api_wait_time
    max_refine_temperature = 0.4 # maximal value of refine temperature

    # test log
    test_log_file_path = args.logdir + "/test_log.txt"
    with open(test_log_file_path, "w") as f:
        if args.domain == "blocksworld":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " blocks." +"\n")
        elif args.domain == "ballmoving":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " balls." +"\n")
        elif args.domain == "cooking":
            f.write("Test log for "+ args.domain + " with " + str(args.num_objects) + " pots." +"\n")


    if args.method == "LLM_trans_no_feedback":

        # if no feedback, use the output directly, same method but no feedback is generated
        max_num_refine = 0
        test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_trans_exact_feedback":

        test_LLM_trans_exact_feedback(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_trans_self_feedback":

        test_LLM_trans_self_feedback(args.domain, test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    elif args.method == "LLM_no_trans":

        max_num_refine = 10
        test_LLM_no_trans(test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time, print_states = args.print_states)

    elif args.method == "LLM_no_trans_simp":
        
        max_num_refine = 10
        test_LLM_no_trans
    
    elif args.method == "LLM_no_trans_self_feedback":

        test_LLM_no_trans_self_feedback(args.domain, test_initial_state, test_goal_state, num_test, max_num_refine, 
                                max_refine_temperature, num_prompt_examples_dataset, test_log_file_path, gpt_api_wait_time)

    else:
        
        raise ValueError("Method not implemented.")





