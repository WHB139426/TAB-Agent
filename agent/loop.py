import re
import json
import shutil
from transformers import Sam3Processor, Sam3Model
from functools import partial

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from agent.prompts import REACT_SYSTEM_PROMPT, SKILL_SYSTEM_PROMPT
from agent.client import Client
from agent.memory import Memory
from agent.skills import SkillLibrary
from agent.tools.base import ToolContext
from agent.tools.registry import ToolRegistry

class AgentLoop:
    def __init__(self, 
                 client_id='/home/haibo/haibo_workspace/weights/Qwen3-VL-32B-Instruct',
                 sam_path='/home/haibo/haibo_workspace/weights/sam3',

                 cache_dir='./react_workspace',
                 skill_dir='./agent/skills',
                 scannet_video_path='/home/haibo/haibo_workspace/data/scannet-frames',
                 scannet_info_path='/home/haibo/haibo_workspace/data/scannet-dataset',

                 max_steps=20,
                 
            ):
        print(f"Loading Client: {client_id}...")
        self.client = Client(client_id)
        print("Loading SAM...")
        self.sam3_processor, self.sam3 = self.initialize_sam_model(sam_path)

        self.cache_dir = cache_dir
        self.scannet_video_path = scannet_video_path
        self.scannet_info_path = scannet_info_path

        self.skill_library = SkillLibrary(skill_dir)
        self.skill_library.load_skills()

        self.tool_context = ToolContext(
            client=self.client,
            sam_processor=self.sam3_processor,
            sam_model=self.sam3,
            cache_dir=self.cache_dir,
            scannet_video_path=self.scannet_video_path,
            scannet_info_path=self.scannet_info_path
        )
        self.tool_registry = ToolRegistry(self.tool_context)

        self.memory = Memory()

        self.max_steps = max_steps

    def to(self, device):
        if self.client.mode == 'local':
            self.client.model.to(device)
        self.sam3.to(device)

    def initialize_sam_model(self, model_name):
        processor = Sam3Processor.from_pretrained(model_name)
        model = Sam3Model.from_pretrained(
            model_name, 
        )
        return processor, model

    def _select_skill(self, query):
        print("🤖 Selecting the best skill for the task...")
        skill_registry = self.skill_library.get_skill_registry_text()        
        if not skill_registry:
            print("Warning: No skills found in library.")
            return None

        selection_prompt = SKILL_SYSTEM_PROMPT.format(
            skill_registry=skill_registry,
            query=query,
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": selection_prompt},],}
        ]
        response = self.client.response(messages)
        selected_skill_name = response.strip().replace("'", "").replace('"', "")
        
        if self.skill_library.has_skill(selected_skill_name):
            print(f"✅ Selected Skill: [{selected_skill_name}]")
            return selected_skill_name
        else:
            print(f"⚠️ LLM hallucinated a skill: {selected_skill_name}. Fallback to none.")
            return None
        
    def _build_system_prompt(self, skill_content):
        tool_desc = self.tool_registry.get_tools_description()
        tool_names = ", ".join(self.tool_registry.get_tool_names())

        if not skill_content:
            skill_content = "No specific expert skill loaded. Please use your general knowledge and tools to solve the problem."

        system_prompt = REACT_SYSTEM_PROMPT.format(
            skill_description=skill_content,
            tool_descriptions=tool_desc,
            tool_names=tool_names,
        )
        return system_prompt

    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", action_text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_args = json.loads(action_match.group(2).strip())
            tool = self.tool_registry.get_tool(tool_name)
            return tool, tool_args
        else:
            return None, None
        
    def _make_cache_dir(self,):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.mkdir(self.cache_dir)

    def run(self, query):

        task_success = False

        selected_skill_name = self._select_skill(query)
        skill_content = self.skill_library.get_skill_description(selected_skill_name)
        sys_prompt = self._build_system_prompt(skill_content)

        self.memory.clear()
        self.memory.add_system_message(sys_prompt)
        self.memory.add_user_message(query)
        
        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")
            response = self.client.response(self.memory.get_chat_history())
            self.memory.add_assistant_message(response)

            thought, action = self._parse_output(response)
            print(f"\nThought: {thought}")
            print(f"\nAction: {action}")

            # case A: SUCCESS (Finish)
            finish_match = re.match(r'^Finish\[(.*)\]$', action, re.DOTALL)
            if finish_match:
                final_answer = finish_match.group(1).strip()
                task_success = True
                print(f"\n✅ Task Finished Successfully. Final Answer: {final_answer}")
                break

            # case B: FAILURE (Abort)
            abort_match = re.match(r'^Abort\[(.*)\]$', action, re.DOTALL)
            if abort_match:
                failure_reason = abort_match.group(1).strip()
                task_success = False
                print(f"\n❌ Task Aborted by Agent. Reason: {failure_reason}")
                break
            
            tool = None
            tool_args = None
            observation = ""

            try:
                tool, tool_args = self._parse_action(f"Action: {action}")
            except Exception as e:
                observation = f"Error parsing action: {str(e)}"
            
            if tool:   
                print(f"\n** Executing Tool: {tool.name} **")
                try:
                    observation = tool.run(**tool_args)
                    print(f"** Tool execution success **")
                except Exception as e:
                    observation = f"Error executing tool {tool.name}: {str(e)}"
            else:
                if not observation:
                    observation = "Error: Invalid Action format. Please use strict format: ToolName(JSON_Args), Finish[result], or Abort[reason]."


            print(f"\nObservation: {observation}")
            self.memory.add_user_message(f"Observation: {observation}")

        save_json(self.memory.get_chat_history(), os.path.join(self.cache_dir, 'chat_history.json'))

        return task_success