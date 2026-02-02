from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import json
import asyncio
import os
import re
from pathlib import Path
from datetime import datetime
from bw_utils import is_image, load_json_file, merge_text_with_limit, create_dir, save_json_file
from BookWorld import BookWorld
from modules.eval_agent import EvalAgent

app = FastAPI()
default_icon_path = './frontend/assets/images/default-icon.jpg'
config = load_json_file('config.json')
for key in config:
    if "API_KEY" in key and config[key]:
        os.environ[key] = config[key]

static_file_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend'))
app.mount("/frontend", StaticFiles(directory=static_file_abspath), name="frontend")

# 预设文件目录
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_presets')

def path_to_url(file_path):
    if not file_path:
        return default_icon_path
    
    # If it's already a URL or starts with /frontend or /data, return as is
    if isinstance(file_path, str) and file_path.startswith(('http', '/frontend', '/data')):
        return file_path
    
    try:
        abs_path = os.path.abspath(file_path)
        root_path = os.path.abspath(os.path.dirname(__file__))
        
        if abs_path.startswith(root_path):
            rel_path = os.path.relpath(abs_path, root_path)
            # Normalize to forward slashes for URLs
            rel_path = rel_path.replace('\\', '/')
            return '/' + rel_path
    except Exception:
        pass
    
    return file_path

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}  
        self.story_tasks: dict[str, asyncio.Task] = {}  
        self.eval_agent = None
        if True:
            if "preset_path" in config and config["preset_path"]:
                if os.path.exists(config["preset_path"]):
                    preset_path = config["preset_path"]
                else:
                    raise ValueError(f"The preset path {config['preset_path']} does not exist.")
            elif "genre" in config and config["genre"]:
                genre = config["genre"]
                preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"./config/experiment_{genre}.json")
            else:
                raise ValueError("Please set the preset_path in `config.json`.")
            self.bw = BookWorld(preset_path = preset_path,
                    world_llm_name = config["world_llm_name"],
                    role_llm_name = config["role_llm_name"],
                    embedding_name = config["embedding_model_name"])
            self.bw.set_generator(rounds = config["rounds"], 
                        save_dir = config["save_dir"], 
                        if_save = config["if_save"],
                        mode = config["mode"],
                        scene_mode = config["scene_mode"],)
        else:
            from BookWorld_test import BookWorld_test
            self.bw = BookWorld_test()
          
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        self.stop_story(client_id)
            
    def stop_story(self, client_id: str):
        if client_id in self.story_tasks:
            self.story_tasks[client_id].cancel()
            del self.story_tasks[client_id]

    async def start_story(self, client_id: str):
        if client_id in self.story_tasks:
            # 如果已经有任务在运行，先停止它
            self.stop_story(client_id)
        
        # 创建新的故事任务
        self.story_tasks[client_id] = asyncio.create_task(
            self.generate_story(client_id)
        )

    async def generate_story(self, client_id: str):
        """持续生成故事的协程"""
        try:
            while True:
                if client_id in self.active_connections:
                    message,status = await self.get_next_message()
                    if message is None:
                        # 故事生成结束
                        await self.active_connections[client_id].send_json({
                            'type': 'story_ended',
                            'data': {'message': 'Simulation finished'}
                        })
                        break
                        
                    await self.active_connections[client_id].send_json({
                        'type': 'message',
                        'data': message
                    })
                    await self.active_connections[client_id].send_json({
                        'type': 'status_update',
                        'data': status
                    })
                    # 添加延迟，控制消息发送频率
                    await asyncio.sleep(0.2)  # 可以调整这个值
                else:
                    break
        except asyncio.CancelledError:
            # 任务被取消时的处理
            print(f"Story generation cancelled for client {client_id}")
        except Exception as e:
            print(f"Error in generate_story: {e}")

    def process_status_data(self, status):
        """处理状态数据中的路径"""
        if 'characters' in status:
            for char in status['characters']:
                if 'icon' in char:
                    char['icon'] = path_to_url(char['icon'])
        return status

    async def get_initial_data(self):
        """获取初始化数据"""
        data = {
            'characters': self.bw.get_characters_info(),
            'map': self.bw.get_map_info(),
            'settings': self.bw.get_settings_info(),
            'status': self.bw.get_current_status(),
            'history_messages':self.bw.get_history_messages(save_dir = config["save_dir"]),
        }
        
        # 处理角色图标路径
        if 'characters' in data:
            for char in data['characters']:
                if 'icon' in char:
                    char['icon'] = path_to_url(char['icon'])
        
        # 处理状态数据中的路径
        if 'status' in data:
            data['status'] = self.process_status_data(data['status'])
                        
        # 处理历史消息图标路径
        if 'history_messages' in data:
            for msg in data['history_messages']:
                if 'icon' in msg:
                    msg['icon'] = path_to_url(msg['icon'])
                        
        return data

    async def get_next_message(self):
        """从BookWorld获取下一条消息"""
        message = self.bw.generate_next_message()
        if message is None:
            return None, None
            
        if not os.path.exists(message["icon"]) or not is_image(message["icon"]):
            message["icon"] = default_icon_path
        else:
            message["icon"] = path_to_url(message["icon"])
            
        status = self.bw.get_current_status()
        status = self.process_status_data(status)
        return message,status

    async def run_evaluation(self, eval_llm_name: str = None):
        """运行评估逻辑，从 server-eval.py 迁移"""
        if not eval_llm_name:
            eval_llm_name = config.get("eval_llm_name", config.get("world_llm_name"))
        
        # 获取对话历史并提取文本
        # manager.bw.server 是 BookWorld.py 中的 Server 实例
        history_texts = self.bw.server.history_manager.get_complete_history()
        
        # 获取角色信息
        characters = self.bw.get_characters_info()
        roles_info = {}
        for char in characters:
            roles_info[char["name"]] = {
                "nickname": char["name"],
                "profile": char["description"]
            }
        
        # 初始化 EvalAgent
        # 注意：EvalAgent 需要模型实例
        self.eval_agent = EvalAgent(
            roles_info=roles_info,
            summary=self.bw.server.script if self.bw.server.script else self.bw.server.intervention,
            source=self.bw.server.source,
            llm_name=eval_llm_name,
            role_llm=self.bw.server.role_llm
        )
        
        mode = config.get("mode", "free")
        language = config.get("language", "zh")
        
        # 按照 server-eval.py 的逻辑进行处理
        start_idx = len(roles_info) if mode == "script" else 2 * len(roles_info)
        max_words = 5000 if language == 'zh' else 3000
        
        text = merge_text_with_limit(text_list=history_texts[start_idx:], max_words=max_words, language=language)
        num_records = len(history_texts) - start_idx
        
        print(f"Starting evaluation with {num_records} records...")
        self.eval_agent.save_generated_text("bookworld", text)
        
        # 1. 评分 BookWorld 生成的内容
        score_result = self.eval_agent.naive_score(text, method="bookworld", mode=mode)
        
        # 2. 生成基准 (Naive) 并对比
        naive_text = self.eval_agent.naive_generate(num_records=num_records, mode=mode)
        self.eval_agent.naive_score(naive_text, method="naive", mode=mode)
        self.eval_agent.naive_winner(naive_text, method="naive", mode=mode)
        
        # 3. 生成多轮基准 (Naive Multi-round) 并对比
        naive_multi_text = self.eval_agent.naive_generate_multi_round(num_records=num_records, mode=mode)
        self.eval_agent.naive_score(naive_multi_text, method="naive_multi", mode=mode)
        self.eval_agent.naive_winner(naive_multi_text, method="naive_multi", mode=mode)
        
        # 4. 尝试 HoLLMwood 对比 (如果适用)
        try:
            pattern = r'(?<=experiment\_)(.*?)(?=\.json)'
            matches = re.findall(pattern, os.path.basename(self.bw.server.preset_path), re.DOTALL)
            if matches:
                idx = matches[0]
                # 假设 root_dir 在这里是 experiment_presets 或者其他
                # 这里简化处理，仅尝试调用
                hollmwood_text = self.eval_agent.hollmwood_generate(
                    root_dir=os.path.dirname(self.bw.server.preset_path),
                    idx=idx,
                    num_records=num_records,
                    default_text=naive_multi_text
                )
                self.eval_agent.naive_score(hollmwood_text, method="hollmwood", mode=mode)
                self.eval_agent.naive_winner(hollmwood_text, method="hollmwood", mode=mode)
        except Exception as e:
            print(f"HoLLMwood evaluation skipped: {e}")
        
        # 保存结果
        experiment_name = self.bw.server.experiment_name
        start_time = self.bw.server.start_time
        role_llm_name = self.bw.server.role_llm_name
        subexperiment_name = config.get("subexperiment_name", "full")
        
        # 提取 source
        world_file_path = self.bw.server.config.get("world_file_path", "")
        source_name = Path(world_file_path).parent.name if world_file_path else "unknown"
        
        save_dir = f"./eval_saves/{mode}/{role_llm_name}/{subexperiment_name}/{source_name}/{experiment_name}/{start_time}"
        save_dir = save_dir.replace("\\", "/")
        create_dir(save_dir)
        
        # 持久化 EvalAgent 状态
        self.eval_agent.save_to_file(save_dir)
        
        # 保存元数据和服务器状态
        meta_info = {
            "location_setted": True,
            "goal_setted": True,
            "round": self.bw.server.cur_round,
            "sub_round": 0,
            "stage": "eval"
        }
        save_json_file(os.path.join(save_dir, "meta_info.json"), meta_info)
        save_json_file(os.path.join(save_dir, "server_info.json"), self.bw.server.__getstate__())
        
        return {
            "success": True,
            "save_dir": save_dir,
            "score": score_result
        }

manager = ConnectionManager()

@app.get("/")
async def get():
    html_file = Path("index.html")
    return HTMLResponse(html_file.read_text(encoding="utf-8"))

# 修复 Vite 注入导致的 404 错误
@app.get("/@vite/client")
async def vite_client():
    return HTMLResponse(content="", status_code=204)

@app.get("/data/{full_path:path}")
async def get_file(full_path: str):
    # 使用项目根目录下的 data 目录作为基础路径
    base_path = Path(__file__).parent / "data"
    file_path = base_path / full_path
    
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    
    # 如果没找到，尝试在根目录下直接查找
    file_path_alt = Path(__file__).parent / full_path
    if file_path_alt.exists() and file_path_alt.is_file():
        return FileResponse(file_path_alt)
        
    return FileResponse(default_icon_path)

@app.get("/api/list-presets")
async def list_presets():
    try:
        # 获取所有json文件
        presets = [f for f in os.listdir(PRESETS_DIR) if f.endswith('.json')]
        return {"presets": presets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-preset")
async def load_preset(request: Request):
    try:
        data = await request.json()
        preset_name = data.get('preset')
        
        if not preset_name:
            raise HTTPException(status_code=400, detail="No preset specified")
            
        preset_path = os.path.join(PRESETS_DIR, preset_name)
        print(f"Loading preset from: {preset_path}")
        
        if not os.path.exists(preset_path):
            raise HTTPException(status_code=404, detail=f"Preset not found: {preset_path}")
            
        try:
            # 更新BookWorld实例的预设
            manager.bw = BookWorld(
                preset_path=preset_path,
                world_llm_name=config["world_llm_name"],
                role_llm_name=config["role_llm_name"],
                embedding_name=config["embedding_model_name"]
            )
            manager.bw.set_generator(
                rounds=config["rounds"],
                save_dir=config["save_dir"],
                if_save=config["if_save"],
                mode=config["mode"],
                scene_mode=config["scene_mode"]
            )
            
            # 获取初始数据
            initial_data = await manager.get_initial_data()
            
            return {
                "success": True,
                "data": initial_data
            }
        except Exception as e:
            print(f"Error initializing BookWorld: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error initializing BookWorld: {str(e)}")
            
    except Exception as e:
        print(f"Error in load_preset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        initial_data = await manager.get_initial_data()
        await websocket.send_json({
            'type': 'initial_data',
            'data': initial_data
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'user_message':
                # 处理用户消息
                record_id = manager.bw.handle_user_message(message['text'])
                # 发送回执给所有客户端，确保记录ID一致
                await websocket.send_json({
                    'type': 'message',
                    'data': {
                        'username': 'User',
                        'timestamp': message['timestamp'],
                        'text': message['text'],
                        'icon': default_icon_path,
                        'uuid': record_id
                    }
                })
                
            elif message['type'] == 'control':
                # 处理控制命令
                if message['action'] == 'start':
                    # Check if rounds is specified in the message
                    rounds = message.get('rounds', config.get('rounds', 10))
                    # Reconfigure generator with specified rounds
                    manager.bw.set_generator(
                        rounds=rounds,
                        save_dir=config.get("save_dir", ""),
                        if_save=config.get("if_save", 0),
                        mode=config.get("mode", "free"),
                        scene_mode=config.get("scene_mode", 1)
                    )
                    await manager.start_story(client_id)
                elif message['action'] == 'pause':
                    manager.stop_story(client_id)
                elif message['action'] == 'stop':
                    manager.stop_story(client_id)
                    # 可以在这里添加额外的停止逻辑
                    
            elif message['type'] == 'edit_message':
                # 处理消息编辑
                edit_data = message['data']
                # 假设 BookWorld 类有一个处理编辑的方法
                manager.bw.handle_message_edit(
                    record_id=edit_data['uuid'],
                    new_text=edit_data['text']
                )
                
            elif message['type'] == 'request_scene_characters':
                manager.bw.select_scene(message['scene'])
                scene_characters = manager.bw.get_characters_info()
                await websocket.send_json({
                    'type': 'scene_characters',
                    'data': scene_characters
                })
                
            elif message['type'] == 'generate_story':
                # 生成故事文本
                story_text = manager.bw.generate_story()
                # 发送生成的故事作为新消息
                await websocket.send_json({
                    'type': 'message',
                    'data': {
                        'username': 'System',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'text': story_text,
                        'icon': default_icon_path,
                        'type': 'story'
                    }
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(client_id)

@app.post("/api/save-config")
async def save_config(request: Request):
    global config
    global manager
    try:
        config_data = await request.json()
        # 检查必要字段是否存在
        if 'provider' not in config_data or 'model' not in config_data or 'apiKey' not in config_data:
            raise HTTPException(status_code=400, detail="缺少必要的字段")

        llm_provider = config_data['provider']
        model = config_data['model']
        api_key = config_data['apiKey']
        config['role_llm_name'] = model
        config['world_llm_name'] = model
        if 'openai' in llm_provider.lower():
            os.environ['OPENAI_API_KEY'] = api_key
        elif 'anthropic' in llm_provider.lower():
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif 'alibaba' in llm_provider.lower():
            os.environ['DASHSCOPE_API_KEY'] = api_key
        elif 'openrouter' in llm_provider.lower():
            os.environ['OPENROUTER_API_KEY'] = api_key
            
        manager.bw.server.reset_llm(model,model)
        return {"status": "success", "message": llm_provider + " 配置已保存"}
    
    except Exception as e:
        print(f"保存配置失败: {e}")
        raise HTTPException(status_code=500, detail="保存配置失败")

@app.get("/api/memory-stats")
async def get_memory_stats():
    """Get memory statistics for all agents and the server."""
    try:
        # Dynamically get save_dir from current simulation state
        # Path format: ./experiment_saves/{experiment_name}/{role_llm_name}_{start_time}
        # Where experiment_name = {preset_basename}/{experiment_subname}_{role_llm_name}
        server = manager.bw.server
        if hasattr(server, 'if_save') and server.if_save and hasattr(server, 'experiment_name'):
            save_dir = f"./experiment_saves/{server.experiment_name}/{server.role_llm_name}_{server.start_time}"
        else:
            save_dir = config.get("save_dir", "")
        
        # Get server history file size
        server_history_file = os.path.join(save_dir, "simulation_history.json") if save_dir else None
        server_short_term_file_size = 0
        if server_history_file and os.path.exists(server_history_file):
            server_short_term_file_size = os.path.getsize(server_history_file)
        
        stats = {
            "current_scene": manager.bw.server.cur_round if hasattr(manager.bw.server, 'cur_round') else 0,
            "save_dir": save_dir,
            "server": {
                "short_term_file_size": server_short_term_file_size,
            },
            "agents": {}
        }
        
        for role_code, agent in manager.bw.server.role_agents.items():
            # Get agent's saved file size (short-term memory)
            agent_file = os.path.join(save_dir, f"roles/{role_code}.json") if save_dir else None
            short_term_file_size = 0
            if agent_file and os.path.exists(agent_file):
                short_term_file_size = os.path.getsize(agent_file)
            
            # Get long-term memory database size
            long_term_db_size = 0
            long_term_record_count = 0
            try:
                # For RoleMemory with ChromaDB
                if hasattr(agent.memory, 'db') and agent.memory.db is not None:
                    db = agent.memory.db
                    db_name = agent.memory.db_name
                    if hasattr(db, 'collections') and db_name in db.collections:
                        collection = db.collections[db_name]
                        long_term_record_count = collection.count()
                        # Estimate size: get all documents and measure
                        all_data = collection.get()
                        if all_data and 'documents' in all_data:
                            docs_text = ''.join(all_data['documents']) if all_data['documents'] else ''
                            long_term_db_size = len(docs_text.encode('utf-8'))
                # For RoleMemory_GA with FAISS/LangChain
                elif hasattr(agent.memory, 'memory_retriever'):
                    retriever = agent.memory.memory_retriever
                    if hasattr(retriever, 'vectorstore') and hasattr(retriever.vectorstore, 'index'):
                        long_term_record_count = retriever.vectorstore.index.ntotal
            except Exception as e:
                print(f"Error getting long-term memory stats for {role_code}: {e}")
            
            agent_stats = {
                "short_term_file_size": short_term_file_size,
                "long_term_record_count": long_term_record_count,
                "long_term_db_size": long_term_db_size,
            }
            stats["agents"][role_code] = agent_stats
        
        return stats
    except Exception as e:
        print(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memory stats: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_simulation(request: Request):
    try:
        data = await request.json()
        eval_llm_name = data.get('eval_llm')
        
        result = await manager.run_evaluation(eval_llm_name=eval_llm_name)
        return result
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
