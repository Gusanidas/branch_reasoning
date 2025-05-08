#!/usr/bin/env python
import asyncio
import os
import argparse
import sys
import time
import signal
import traceback
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from branch_reasoning.generation.vllm_generation import (
    start_vllm, 
    stop_vllm, 
    kill_vllm_workers,
    vllm_chat_generate_text,
    vllm_generate_text,
    vLLM
)

# Global variable to track the vLLM instance for cleanup during signals
current_vllm_instance: Optional[vLLM] = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print(f"\n\nReceived signal {sig}, shutting down vLLM...")
    if current_vllm_instance is not None:
        # We need to run the async cleanup in a new event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup_vllm(current_vllm_instance))
        finally:
            loop.close()
    
    # Force kill any remaining vLLM workers to be safe
    kill_vllm_workers()
    print("All vLLM processes terminated, exiting...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

async def cleanup_vllm(vllm_instance: vLLM):
    """
    Cleanup function to ensure vLLM resources are properly released.
    """
    try:
        await stop_vllm(vllm_instance)
        print("vLLM server stopped successfully during cleanup")
    except Exception as e:
        print(f"Error during vLLM cleanup: {e}")
        # Ensure workers are killed even if stop_vllm fails
        kill_vllm_workers()

async def main(
    model_name: str = "Qwen/Qwen2.5-1B-Instruct",
    local_dir: str = "./saved_model",
    use_chat_api: bool = True,
    prompt: str = "Explain reinforcement learning in a few sentences",
    system_message: str = "You are a helpful AI assistant.",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_completions: int = 1,
    use_bfloat16: bool = False,
    verbose: bool = True,
):
    """
    Test script for vLLM - download a model from Hugging Face, save it locally, 
    start a vLLM server with it, and generate some text.
    
    Args:
        model_name: Name of the model on Hugging Face to use
        local_dir: Local directory to save the model to
        use_chat_api: Whether to use the chat API or completion API
        prompt: Text prompt for generation
        system_message: System message for chat completions
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p value for sampling
        num_completions: Number of completions to generate
        use_bfloat16: Whether to use bfloat16 precision
        verbose: Whether to print verbose output
    """
    global current_vllm_instance
    vllm_instance = None
    
    try:
        # Step 1: Download and save the model locally
        if verbose:
            print(f"Downloading model: {model_name}")
        
        # Determine dtype based on argument
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        
        # Download and save the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype
        )
        
        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"  # This is important for vLLM
        )
        
        # If the tokenizer doesn't have a pad token, use the eos token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Save model and tokenizer
        if verbose:
            print(f"Saving model and tokenizer to: {local_dir}")
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        
        # Free memory after saving
        del model
        torch.cuda.empty_cache()
        
        # Step 2: Start vLLM server
        if verbose:
            print("Starting vLLM server...")
        
        # vLLM server arguments
        vllm_server_args = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 2048,
            "disable_log_requests": False,
            "enable_prefix_caching": True,
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("./logs", exist_ok=True)
        
        # Start the vLLM server
        vllm_instance = await start_vllm(
            model=local_dir,
            log_file="./logs/test_vllm.log",
            max_concurrent_requests=8,
            named_arguments=vllm_server_args,
            verbosity=2 if verbose else 1,
            timeout=180.0,  # 3 minutes timeout
        )
        
        # Set the global instance for signal handling
        current_vllm_instance = vllm_instance
        
        # Step 3: Generate text
        generation_args = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if verbose:
            print(f"\nGenerating {'chat' if use_chat_api else 'completion'} with prompt:\n{prompt}\n")
        
        # Generate text using the appropriate function
        start_time = time.time()
        if use_chat_api:
            completions = await vllm_chat_generate_text(
                vllm_instance=vllm_instance,
                prompts=[prompt],
                num_completions=num_completions,
                generation_args=generation_args,
                system_message=system_message
            )
        else:
            completions = await vllm_generate_text(
                vllm_instance=vllm_instance,
                prompts=[prompt],
                num_completions=num_completions,
                generation_args=generation_args
            )
        
        generation_time = time.time() - start_time
        
        # Print generated completions
        print("\n" + "="*50)
        print(f"Generated {len(completions)} completion(s) in {generation_time:.2f}s:")
        for i, completion in enumerate(completions):
            print(f"\n--- Completion {i+1} ---")
            print(completion)
        print("="*50)
        
        # Step 4: Clean up
        if verbose:
            print("\nStopping vLLM server...")
        await stop_vllm(vllm_instance)
        if verbose:
            print("vLLM server stopped successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources if needed
        if vllm_instance is not None and vllm_instance.process.returncode is None:
            try:
                await stop_vllm(vllm_instance)
            except Exception:
                pass
        
        # Reset the global instance
        current_vllm_instance = None
        
        # Make sure to clean up any remaining processes
        kill_vllm_workers()
        
        if verbose:
            print("All cleanup complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM by loading a model and generating text")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1B-Instruct", help="Hugging Face model to load")
    parser.add_argument("--local-dir", default="./saved_model", help="Local directory to save the model")
    parser.add_argument("--prompt", default="Explain reinforcement learning in a few sentences", help="Prompt for text generation")
    parser.add_argument("--system-message", default="You are a helpful AI assistant.", help="System message for chat completions")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value for generation")
    parser.add_argument("--num-completions", type=int, default=1, help="Number of completions to generate")
    parser.add_argument("--use-completion", action="store_true", help="Use completion API instead of chat API")
    parser.add_argument("--use-bfloat16", action="store_true", help="Use bfloat16 precision instead of float16")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        # Run the async main function
        asyncio.run(main(
            model_name=args.model,
            local_dir=args.local_dir,
            use_chat_api=not args.use_completion,
            prompt=args.prompt,
            system_message=args.system_message,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_completions=args.num_completions,
            use_bfloat16=args.use_bfloat16,
            verbose=not args.quiet,
        ))
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    finally:
        # Ensure all vLLM processes are cleaned up
        kill_vllm_workers()
        print("Final cleanup complete")