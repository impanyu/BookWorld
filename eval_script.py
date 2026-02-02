#!/usr/bin/env python3
"""
BookWorld Evaluation Script

This script runs simulation rounds and then triggers the evaluation API.
Make sure the server is running before executing this script.

Usage:
    python eval_script.py                                    # Run 10 rounds then evaluate
    python eval_script.py --rounds 15 --eval_llm gpt-4o      # Run 15 rounds with specific eval model
    python eval_script.py --skip_simulation                  # Skip simulation, only evaluate
"""

import argparse
import asyncio
import requests
import json
import sys
import uuid
import os
from datetime import datetime

try:
    import websockets
except ImportError:
    print("ERROR: websockets library required. Install with: pip install websockets")
    sys.exit(1)


def get_memory_stats(base_url: str = "http://localhost:8000") -> dict:
    """Fetch memory statistics from the server."""
    try:
        response = requests.get(f"{base_url}/api/memory-stats", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Warning: Could not fetch memory stats: {e}")
        return None


async def run_simulation(base_url: str = "http://localhost:8000",
                         rounds: int = 10,
                         timeout: int = 600,
                         stats_output_file: str = "memory_stats.json") -> tuple:
    """
    Run simulation via WebSocket connection and collect memory statistics.
    
    Args:
        base_url: Base URL of the BookWorld server
        rounds: Number of simulation rounds to wait for
        timeout: Maximum time to wait in seconds
        stats_output_file: Path to save memory statistics JSON
    
    Returns:
        tuple: (success: bool, memory_stats: dict)
    """
    # Storage for memory statistics per scene
    memory_stats_history = {
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "requested_rounds": rounds,
            "base_url": base_url
        },
        "scenes": []
    }
    # Convert http to ws
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    client_id = str(uuid.uuid4())
    ws_endpoint = f"{ws_url}/ws/{client_id}"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to WebSocket...")
    print(f"  URL: {ws_endpoint}")
    
    try:
        async with websockets.connect(ws_endpoint, ping_interval=30, ping_timeout=10) as websocket:
            # Wait for initial data
            initial_msg = await asyncio.wait_for(websocket.recv(), timeout=30)
            initial_data = json.loads(initial_msg)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected! Received initial data.")
            
            # Start simulation with specified rounds
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting simulation ({rounds} rounds)...")
            start_command = json.dumps({
                "type": "control",
                "action": "start",
                "rounds": rounds
            })
            await websocket.send(start_command)
            
            # Listen for messages until simulation ends or timeout
            message_count = 0
            scene_count = 0
            start_time = datetime.now()
            simulation_ended = False
            
            print("-" * 50)
            print("Simulation running... (Ctrl+C to interrupt)")
            
            while True:
                try:
                    # Check timeout
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > timeout:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Timeout reached ({timeout}s)")
                        break
                    
                    # Wait for message with timeout
                    remaining = timeout - elapsed
                    msg = await asyncio.wait_for(websocket.recv(), timeout=min(remaining, 120))
                    data = json.loads(msg)
                    message_count += 1
                    
                    msg_type = data.get('type', '')
                    
                    # Check if simulation ended
                    if msg_type == 'story_ended':
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Server signaled simulation complete!")
                        simulation_ended = True
                        break
                    
                    # Track progress for regular messages
                    elif msg_type == 'message':
                        msg_data = data.get('data', {})
                        username = msg_data.get('username', 'Unknown')
                        text = msg_data.get('text', '')[:80] if msg_data.get('text') else ''
                        scene = msg_data.get('scene', '')
                        
                        # Track scenes and collect memory stats on scene change
                        if scene:
                            try:
                                scene_num = int(scene)
                                if scene_num > scene_count:
                                    # Scene changed - collect memory stats for previous scene
                                    if scene_count >= 0:
                                        stats = get_memory_stats(base_url)
                                        if stats:
                                            stats["scene_number"] = scene_count
                                            stats["timestamp"] = datetime.now().isoformat()
                                            memory_stats_history["scenes"].append(stats)
                                            print(f" [Stats collected for scene {scene_count}]")
                                    
                                    scene_count = scene_num
                                    print(f"\n  [Scene {scene_count + 1}]")
                            except (ValueError, TypeError):
                                pass
                        
                        # Print message preview
                        if text:
                            print(f"    {username}: {text}...")
                    
                    elif msg_type == 'status_update':
                        # Status updates are normal, just count them
                        pass
                        
                    elif msg_type == 'system':
                        text = data.get('data', {}).get('text', str(data))[:100]
                        print(f"  [System] {text}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue waiting
                    print(".", end="", flush=True)
                    continue
            
            # Collect final memory stats
            final_stats = get_memory_stats(base_url)
            if final_stats:
                final_stats["scene_number"] = scene_count
                final_stats["timestamp"] = datetime.now().isoformat()
                final_stats["is_final"] = True
                memory_stats_history["scenes"].append(final_stats)
                print(f" [Final stats collected for scene {scene_count}]")
            
            print("-" * 50)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Simulation finished!")
            print(f"  Total messages: {message_count}")
            print(f"  Scenes completed: {scene_count + 1}")
            print(f"  Ended normally: {simulation_ended}")
            
            # Add summary to stats
            memory_stats_history["metadata"]["end_time"] = datetime.now().isoformat()
            memory_stats_history["metadata"]["total_messages"] = message_count
            memory_stats_history["metadata"]["scenes_completed"] = scene_count + 1
            memory_stats_history["metadata"]["simulation_ended_normally"] = simulation_ended
            
            # Save memory stats to JSON file
            try:
                with open(stats_output_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_stats_history, f, indent=2, ensure_ascii=False)
                print(f"  Memory stats saved to: {stats_output_file}")
            except Exception as e:
                print(f"  Warning: Could not save memory stats: {e}")
            
            # Send stop command
            stop_command = json.dumps({
                "type": "control",
                "action": "stop"
            })
            await websocket.send(stop_command)
            
            return True, memory_stats_history
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"ERROR: WebSocket connection closed: {e}")
        return False, memory_stats_history
    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to server at {base_url}")
        print("Make sure the BookWorld server is running: python server.py")
        return False, memory_stats_history
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}")
        return False, memory_stats_history


def run_evaluation(base_url: str = "http://localhost:8000", 
                   eval_llm: str = None,
                   timeout: int = 600) -> dict:
    """
    Trigger the BookWorld evaluation API.
    
    Args:
        base_url: Base URL of the BookWorld server
        eval_llm: LLM model to use for evaluation (e.g., 'gpt-4o', 'claude-3.5-sonnet')
        timeout: Request timeout in seconds (default 600s = 10 minutes)
    
    Returns:
        dict: Evaluation results from the API
    """
    url = f"{base_url}/api/evaluate"
    
    # Build payload
    payload = {}
    if eval_llm:
        payload["eval_llm"] = eval_llm
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation...")
    print(f"  URL: {url}")
    print(f"  Eval LLM: {eval_llm or '(default from config)'}")
    print(f"  Timeout: {timeout}s")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluation completed!")
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to server at {base_url}")
        print("Make sure the BookWorld server is running:")
        print("  python server.py")
        sys.exit(1)
        
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timed out after {timeout} seconds")
        print("Evaluation may take a long time. Try increasing --timeout")
        sys.exit(1)
        
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        if response.text:
            print(f"Response: {response.text}")
        sys.exit(1)
        
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse response as JSON")
        print(f"Response: {response.text}")
        sys.exit(1)


def print_results(result: dict):
    """Pretty print the evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    if isinstance(result, dict):
        # Check for common result fields
        if "status" in result:
            print(f"Status: {result['status']}")
        
        if "message" in result:
            print(f"Message: {result['message']}")
        
        if "scoring" in result:
            print("\n--- Scoring ---")
            for dimension, score in result["scoring"].items():
                print(f"  {dimension}: {score}")
        
        if "winner" in result:
            print(f"\n--- Winner Comparison ---")
            print(f"  {result['winner']}")
        
        if "save_path" in result:
            print(f"\nResults saved to: {result['save_path']}")
        
        # Print full result as JSON for debugging
        print("\n--- Full Response ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)


def main():
    parser = argparse.ArgumentParser(
        description="BookWorld Evaluation Script - Runs simulation and evaluates results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_script.py                                    # Run 10 rounds then evaluate
  python eval_script.py --rounds 15                        # Run 15 rounds then evaluate  
  python eval_script.py --rounds 10 --eval_llm gpt-4o      # Specify eval model
  python eval_script.py --skip_simulation                  # Only evaluate (skip simulation)
  python eval_script.py --url http://192.168.1.100:8000    # Custom server URL

Best Practices:
  - Run at least 10 simulation rounds before evaluating
  - Use gpt-4o or claude-3.5-sonnet for accurate evaluation
  - Don't run multiple evaluations concurrently
        """
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of simulation rounds to run (default: 10)"
    )
    
    parser.add_argument(
        "--skip_simulation",
        action="store_true",
        help="Skip simulation and only run evaluation"
    )
    
    parser.add_argument(
        "--eval_llm", 
        type=str, 
        default=None,
        help="LLM model for evaluation (e.g., gpt-4o, claude-3.5-sonnet). Default: uses config.json"
    )
    
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8000",
        help="BookWorld server URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=600,
        help="Timeout in seconds for simulation and evaluation (default: 600)"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Only output the JSON result"
    )
    
    parser.add_argument(
        "--stats_file",
        type=str,
        default="memory_stats.json",
        help="Output file for memory statistics (default: memory_stats.json)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Run simulation (unless skipped)
    memory_stats = None
    if not args.skip_simulation:
        print("=" * 60)
        print("STEP 1: RUNNING SIMULATION")
        print("=" * 60)
        
        success, memory_stats = asyncio.run(run_simulation(
            base_url=args.url,
            rounds=args.rounds,
            timeout=args.timeout,
            stats_output_file=args.stats_file
        ))
        
        if not success:
            print("\nERROR: Simulation failed. Aborting evaluation.")
            sys.exit(1)
        
        print()
    
    # Step 2: Run evaluation
    print("=" * 60)
    print("STEP 2: RUNNING EVALUATION")
    print("=" * 60)
    
    if args.quiet:
        # Quiet mode - only output JSON
        result = run_evaluation(
            base_url=args.url,
            eval_llm=args.eval_llm,
            timeout=args.timeout
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Normal mode with pretty output
        result = run_evaluation(
            base_url=args.url,
            eval_llm=args.eval_llm,
            timeout=args.timeout
        )
        print_results(result)
    
    # Step 3: Move memory_stats.json to eval output folder
    if memory_stats and result and isinstance(result, dict) and result.get("save_dir"):
        eval_save_dir = result["save_dir"]
        final_stats_path = os.path.join(eval_save_dir, "memory_stats.json")
        try:
            # Save memory stats to eval output folder
            with open(final_stats_path, 'w', encoding='utf-8') as f:
                json.dump(memory_stats, f, indent=2, ensure_ascii=False)
            print(f"\nMemory stats saved to: {final_stats_path}")
            
            # Remove the temporary file in root directory
            if os.path.exists(args.stats_file) and args.stats_file != final_stats_path:
                os.remove(args.stats_file)
                print(f"Removed temporary file: {args.stats_file}")
        except Exception as e:
            print(f"Warning: Could not save memory stats to eval folder: {e}")


if __name__ == "__main__":
    main()
