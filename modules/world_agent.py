import sys
sys.path.append("../")
import csv
from typing import Any, Dict, List, Optional, Literal
from bw_utils import *
from modules.embedding import get_embedding_model
from modules.history_manager import HistoryManager

class WorldAgent:
    # Init
    def __init__(self, 
                 world_file_path: str,
                 location_file_path: str,
                 map_file_path: Optional[str] = "",
                 world_description: str = "",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None,
                 db_type: str = "chroma",
                 language: str = "zh",
                 ):
        if llm is None:
            llm = get_models(llm_name)
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=language)
        self.llm = llm
        self.embedding = embedding
        self.world_info: Dict[str, Any] = load_json_file(world_file_path)
        self.world_name: str = self.world_info["world_name"]
        self.language: str = language
        self.description:str = self.world_info["description"] if world_description == "" else world_description
        source = self.world_info["source"]
        
        self.locations_info: Dict[str, Any] = {}  
        self.locations: List[str] = []
        self.history: List[str] = []
        self.edges: Dict[tuple, int] = {}
        self.prompts: List[Dict] = []

        self.memory = HistoryManager(embedding_fn=embedding)
        
        self.init_from_file(map_file_path = map_file_path,
                            location_file_path = location_file_path)
        self.init_prompt()

            
        self.world_data,self.world_settings = build_world_agent_data(world_file_path = world_file_path, max_words = 50)
        self.db_name = clean_collection_name(f"settings_{source}_{embedding_name}")
        self.db = build_db(data = [row for row in self.world_data], 
                           db_name = self.db_name, 
                           db_type = db_type, 
                           embedding = embedding)
        
    def init_from_file(self, map_file_path: str, location_file_path: str, default_distance: int = 1):
        if map_file_path and os.path.exists(map_file_path):
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            with open(map_file_path, mode='r',encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                locations = next(csv_reader)[1:]  
                for row in csv_reader:
                    loc1 = row[0]
                    if loc1 not in valid_locations:
                        print(f"Warning: The location {loc1} does not exist")
                        continue
                    self.locations_info[loc1] = valid_locations[loc1]
                    self.locations.append(loc1)
                    distances = row[1:]
                    for i, distance in enumerate(distances):
                        loc2 = locations[i]
                        if loc2 not in valid_locations:
                            print(f"Warning: The location {loc2} does not exist")
                            continue
                        if distance != '0':  # Skip self-loops
                            self._add_edge(loc1, loc2, int(distance))
        else:
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            for loc1 in valid_locations:
                self.locations_info[loc1] = valid_locations[loc1]
                self.locations.append(loc1)
                for loc2 in valid_locations:
                    if loc2 != loc1:
                        self._add_edge(loc1, loc2, default_distance)
                        
    def init_prompt(self,):
        if self.language == "zh":
            from modules.prompt.world_agent_prompt_zh import (
                ENVIROMENT_INTERACTION_PROMPT, NPC_INTERACTION_PROMPT,
                SCRIPT_INSTRUCTION_PROMPT, SCRIPT_ATTENTION_PROMPT,
                DECIDE_NEXT_ACTOR_PROMPT, GENERATE_INTERVENTION_PROMPT,
                UPDATE_EVENT_PROMPT, LOCATION_PROLOGUE_PROMPT,
                SELECT_SCREEN_ACTORS_PROMPT, JUDGE_IF_ENDED_PROMPT,
                LOG2STORY_PROMPT,
                CONSENSUS_EXTRACT_PROMPT, CONSENSUS_DEDUP_PROMPT,
                CONSENSUS_FILTER_PROMPT,
            )
        else:
            from modules.prompt.world_agent_prompt_en import (
                ENVIROMENT_INTERACTION_PROMPT, NPC_INTERACTION_PROMPT,
                SCRIPT_INSTRUCTION_PROMPT, SCRIPT_ATTENTION_PROMPT,
                DECIDE_NEXT_ACTOR_PROMPT, GENERATE_INTERVENTION_PROMPT,
                UPDATE_EVENT_PROMPT, LOCATION_PROLOGUE_PROMPT,
                SELECT_SCREEN_ACTORS_PROMPT, JUDGE_IF_ENDED_PROMPT,
                LOG2STORY_PROMPT,
                CONSENSUS_EXTRACT_PROMPT, CONSENSUS_DEDUP_PROMPT,
                CONSENSUS_FILTER_PROMPT,
            )
            
        self._ENVIROMENT_INTERACTION_PROMPT = ENVIROMENT_INTERACTION_PROMPT
        self._NPC_INTERACTION_PROMPT = NPC_INTERACTION_PROMPT
        self._SCRIPT_INSTRUCTION_PROMPT = SCRIPT_INSTRUCTION_PROMPT
        self._SCRIPT_ATTENTION = SCRIPT_ATTENTION_PROMPT
        self._DECIDE_NEXT_ACTOR_PROMPT= DECIDE_NEXT_ACTOR_PROMPT
        self._LOCATION_PROLOGUE_PROMPT = LOCATION_PROLOGUE_PROMPT
        self._GENERATE_INTERVENTION_PROMPT = GENERATE_INTERVENTION_PROMPT
        self._UPDATE_EVENT_PROMPT = UPDATE_EVENT_PROMPT
        self._SELECT_SCREEN_ACTORS_PROMPT = SELECT_SCREEN_ACTORS_PROMPT
        self._JUDGE_IF_ENDED_PROMPT = JUDGE_IF_ENDED_PROMPT
        self._LOG2STORY_PROMPT = LOG2STORY_PROMPT
        self._CONSENSUS_EXTRACT_PROMPT = CONSENSUS_EXTRACT_PROMPT
        self._CONSENSUS_DEDUP_PROMPT = CONSENSUS_DEDUP_PROMPT
        self._CONSENSUS_FILTER_PROMPT = CONSENSUS_FILTER_PROMPT
        
    # Agent
    def update_event(self, 
                     cur_event: str, 
                     intervention:str,
                     history_text: str, 
                     script: str = ""):
        prompt = self._UPDATE_EVENT_PROMPT.format(**{
            "event":cur_event,
            "intervention":intervention,
            "history":history_text
        })
        if script:
            prompt = self._SCRIPT_ATTENTION.format(script = script) + prompt
        new_event = self.llm.chat(prompt)
        self.record(new_event, prompt)
        return new_event
    
    def decide_next_actor(self, 
                          history_text: str, 
                          roles_info_text: str,
                          script: str = "",
                          event: str = "",
                          last_actor_code: str = ""):
        prompt = self._DECIDE_NEXT_ACTOR_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
            "last_actor": last_actor_code if last_actor_code else "无"
        })
        
        max_tries = 3
        for _ in range(max_tries):
            try:
                response = self.llm.chat(prompt)
                break
            except Exception as e:
                print(f"Parsing failure! Error:", e)    
                print(response)
        role_code = response
        self.prompts.append({"prompt":prompt,
                            "response":f"{role_code}"})

        return role_code
    
    def judge_if_ended(self,history_text):
        prompt = self._JUDGE_IF_ENDED_PROMPT.format(**{
            "history":history_text
        })
        max_tries = 3
        response = {"if_end":True, "detail":""}
        for _ in range(max_tries):
            try:
                response.update(json_parser(self.llm.chat(prompt)))
                break
            except Exception as e:
                print(f"Parsing failure! Error:", e)    
                print(response)
        
        return response["if_end"],response["detail"]
        
    def decide_scene_actors(self,roles_info_text, history_text, event, previous_role_codes):
        prompt = self._SELECT_SCREEN_ACTORS_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
            "event":event,
            "previous_role_codes":previous_role_codes
            
        })
        response = self.llm.chat(prompt)
        role_codes = eval(response)
        return role_codes
    
    def generate_location_prologue(self,
                                   location_code,
                                   history_text,
                                   event,
                                   location_info_text):
        prompt = self._LOCATION_PROLOGUE_PROMPT.format(**{
            "location_name":self.locations_info[location_code]["location_name"],
            "location_description":self.locations_info[location_code]["location_name"],
            "location_info":location_info_text,
            "history_text":history_text,
            "event":event,
            "world_description":self.description
        })
        response = self.llm.chat(prompt)
        self.record(detail = response,prompt = prompt)
        return "\n"+response
    
    def enviroment_interact(self, 
                            action_maker_name: str, 
                            action: str,
                            action_detail: str, 
                            location_code: str):
        references = self.retrieve_references(query = action_detail)
        prompt = self._ENVIROMENT_INTERACTION_PROMPT.format(**
            {
                "role_name":action_maker_name,
                "action":action,
                "action_detail":action_detail,
                "world_description":self.description,
                "location":location_code,
                "location_description":self.locations_info[location_code]["detail"],
                "references":references,
            }
            )
        response = "无事发生。" if self.language == "zh" else "Nothing happens."
        for i in range(3):
            try:
                response = self.llm.chat(prompt) 
                if response:
                    break
            except Exception as e:
                print("Enviroment Interaction failed! {i}th tries. Error:", e)
        self.record(response, prompt)
        return response
    
    
    def npc_interact(self, 
                     action_maker_name: str, 
                     action_detail: str, 
                     location_name: str,
                     target_name: str):
        references = self.retrieve_references(query = action_detail)
        prompt = self._NPC_INTERACTION_PROMPT.format(**
            {
                "role_name":action_maker_name,
                "action_detail":action_detail,
                "world_description":self.description,
                "target":target_name,
                "references":references,
                "location":location_name
            }
            )
        
        npc_interaction = {"if_end_interaction":True,"detail":"无事发生。"} if self.language == "zh" else {"if_end_interaction":True,"detail":"Nothing happens"}
        try:
            npc_interaction = json_parser(self.llm.chat(prompt))
            response = npc_interaction["detail"]
            self.record(response, prompt)
        except Exception as e:
            print("Enviroment Interaction failed!",e)
        
        return npc_interaction
    
    
    def get_script_instruction(self, 
                               roles_info_text: str, 
                               event: str, 
                               history_text: str, 
                               script: str, 
                               last_progress: str):
        prompt = self._SCRIPT_INSTRUCTION_PROMPT.format(**{
            "roles_info":roles_info_text,
            "event":event,
            "history_text":history_text,
            "script":script,
            "last_progress":last_progress
        })
        max_tries = 3
        instruction = {}
        for i in range(max_tries):
            response = self.llm.chat(prompt)
            try:
                instruction = json_parser(response)
                break
            except Exception as e:
                print(f"Parsing failure! {i+1}th tries. Error:", e)   
                print(response)
        self.record(response, prompt)
        return instruction
    
    def generate_event(self,roles_info_text: str, event: str, history_text: str):
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        response = self.llm.chat(prompt)
        self.record(response, prompt)
        return response
        
    def generate_script(self, roles_info_text: str, event: str, history_text: str):
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        response = self.llm.chat(prompt)
        self.record(response, prompt)
        return response
    
    def log2story(self,logs):
        prompt = self._LOG2STORY_PROMPT.format(**{
            "logs":logs
        })
        response = self.llm.chat(prompt)
        return response
    
    # ------------------------------------------------------------------
    # Consensus mechanism
    # ------------------------------------------------------------------

    def check_and_run_consensus(self, role_agents: dict, consensus_threshold: int = 10):
        """Run the full consensus pipeline when total unconsensused memories
        across all role agents reach *consensus_threshold*.

        Pipeline:
          1. Collect all unconsensused items from every role agent.
          2. LLM extracts consensus items (skip contradictions → intersection).
          3. For each consensus item, deduplicate against world memory
             (top-5 similar); keep only novel, non-contradictory items.
          4. Add surviving consensus items to world memory → "final consensus".
          5. For each role agent, filter their unconsensused items against
             the final consensus: remove items fully covered by it.
          6. Mark everything as consensused.

        Returns:
            list[str]: The final consensus items added to world memory, or [].
        """
        total_new = sum(
            agent.history_manager.get_unconsensused_count()
            for agent in role_agents.values()
        )
        if total_new < consensus_threshold:
            return []

        # Step 1 — gather unconsensused items
        all_new_memories_text = ""
        agent_unconsensused: Dict[str, list] = {}
        for role_code, agent in role_agents.items():
            items = agent.history_manager.get_unconsensused_items()
            if items:
                agent_unconsensused[role_code] = items
                role_name = getattr(agent, "role_name", role_code)
                mem_lines = "\n".join(f"- {it['detail']}" for it in items)
                all_new_memories_text += f"\n### {role_name}:\n{mem_lines}\n"

        if not all_new_memories_text.strip():
            return []

        # Step 2 — LLM extracts consensus (skip contradictions)
        raw_consensus = self._extract_consensus(all_new_memories_text)
        if not raw_consensus:
            self._mark_all_agents_consensused(role_agents)
            return []

        # Step 3 — deduplicate each consensus item against world memory
        final_consensus: List[str] = []
        for item_text in raw_consensus:
            if self._check_consensus_novelty(item_text):
                self.memory.add_memory(item_text, {"source": "consensus"})
                final_consensus.append(item_text)

        if not final_consensus:
            self._mark_all_agents_consensused(role_agents)
            return []

        # Step 4 — filter each role agent's unconsensused items
        final_consensus_text = "\n".join(f"- {c}" for c in final_consensus)
        for role_code, items in agent_unconsensused.items():
            agent = role_agents[role_code]
            self._filter_agent_memories(agent, items, final_consensus_text)

        # Step 5 — mark everything as consensused
        self._mark_all_agents_consensused(role_agents)

        self.history.append(f"[Consensus] {final_consensus_text}")
        return final_consensus

    # ---- consensus sub-steps ----

    def _extract_consensus(self, all_new_memories: str) -> list:
        """LLM extracts consensus items from all agents' new memories."""
        prompt = self._CONSENSUS_EXTRACT_PROMPT.format(
            all_new_memories=all_new_memories
        )
        try:
            response = self.llm.chat(prompt)
            return self._parse_json_list(response)
        except Exception as e:
            print(f"Consensus extraction failed: {e}")
            return []

    @staticmethod
    def _parse_json_list(text: str) -> list:
        """Parse a JSON list from LLM output, tolerating markdown fences."""
        import re, json as _json
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = _json.loads(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result if item]
            except _json.JSONDecodeError:
                pass
        return []

    def _check_consensus_novelty(self, consensus_item: str) -> bool:
        """Return True if *consensus_item* should be added to world memory
        (novel & non-contradictory vs. existing world memory)."""
        if len(self.memory) == 0:
            return True

        similar = self.memory.retrieve_by_similarity(consensus_item, top_k=5)
        if not similar:
            return True

        existing_context = "\n".join(f"- {s}" for s in similar)
        prompt = self._CONSENSUS_DEDUP_PROMPT.format(
            consensus_item=consensus_item,
            existing_memories=existing_context,
        )
        try:
            response = self.llm.chat(prompt)
            result = json_parser(response)
            return result.get("action", "keep") == "keep"
        except Exception as e:
            print(f"Consensus dedup check failed: {e}")
            return True  # keep on error

    def _filter_agent_memories(
        self, agent, unconsensused_items: list, final_consensus_text: str
    ):
        """Remove agent memory items fully covered by the final consensus."""
        for item_info in unconsensused_items:
            memory_text = item_info["detail"]
            idx = item_info["idx"]

            prompt = self._CONSENSUS_FILTER_PROMPT.format(
                consensus=final_consensus_text,
                memory=memory_text,
            )
            try:
                response = self.llm.chat(prompt)
                result = json_parser(response)
                if result.get("action", "keep") == "remove":
                    agent.history_manager.remove_record(idx)
            except Exception as e:
                print(f"Consensus filter failed for record {idx}: {e}")

    @staticmethod
    def _mark_all_agents_consensused(role_agents: dict):
        for agent in role_agents.values():
            agent.history_manager.mark_all_consensused()

    # ------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------

    def record(self, detail: str, prompt: str = ""):
        if prompt:
            self.prompts.append({"prompt":prompt,
                                 "response":detail})
        self.history.append(detail)
    
    def add_location_during_simulation(self, location: str, detail: str):
        self.locations.append(location)
        self.locations_info[location] = {
            'location_code': location,
            "location_name": location,
            'description': '',
            'detail':detail
        }
        for loc in self.locations:
            if loc != location:
                self._add_edge(loc, location, 1)
                self._add_edge(location,loc, 1)
        return
    
    def retrieve_references(self, query: str, top_k = 3, max_words = 100):
        if self.db is None:
            return ""
        references = "\n".join(self.db.search(query, top_k,self.db_name))
        references = references[:max_words]
        return references

    def find_location_name(self, code: str):
        return self.locations_info[code]["location_name"]
              
    def _add_location(self, code: str, location_info: Dict[str, Any]):
        self.locations_info[code] = location_info
        
    def _add_edge(self, code1: str, code2: str, distance: int):
        self.edges[(code1,code2)] = distance
        self.edges[(code2,code1)] = distance  
        
    def get_distance(self, code1: str, code2: str):
        if (code1,code2) in self.edges:
            return self.edges[(code1,code2)]
        else:
            return None
        
    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() 
                 if isinstance(value, (str, int, list, dict, float, bool, type(None)))
                 and (key not in ['llm','embedding','db','locations_info','edges',
                                  'world_data','world_settings','memory']
                 and "PROMPT" not in key)
                 }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./world_agent.json")
        save_json_file(filename, self.__getstate__() )
        self.memory.save_to_file(root_dir, "world_memory.json")

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./world_agent.json")
        state = load_json_file(filename)
        self.__setstate__(state)
        self.memory.load_from_file(root_dir, "world_memory.json")
        if self.embedding:
            self.memory.set_embedding_fn(self.embedding)
