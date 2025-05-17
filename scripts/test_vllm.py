#!/usr/bin/env python
import asyncio
import os
<<<<<<< HEAD
import argparse
=======
>>>>>>> vllm
import sys
import time
import signal
import traceback
<<<<<<< HEAD
=======
import gc
>>>>>>> vllm
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from branch_reasoning.generation.vllm_generation import (
<<<<<<< HEAD
    start_vllm, 
    stop_vllm, 
=======
    start_vllm,
    stop_vllm,
>>>>>>> vllm
    kill_vllm_workers,
    vllm_chat_generate_text,
    vllm_generate_text,
    vLLM
)

# Global variable to track the vLLM instance for cleanup during signals
current_vllm_instance: Optional[vLLM] = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
<<<<<<< HEAD
    print(f"\n\nReceived signal {sig}, shutting down vLLM...")
=======
    """Signal handler for graceful shutdown."""
    print(f"\n\nReceived signal {sig}, shutting down vLLM...")

>>>>>>> vllm
    if current_vllm_instance is not None:
        # We need to run the async cleanup in a new event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup_vllm(current_vllm_instance))
        finally:
            loop.close()
<<<<<<< HEAD
    
=======

>>>>>>> vllm
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

<<<<<<< HEAD
async def main(
    model_name: str = "Qwen/Qwen2.5-1B-Instruct",
=======
def _print_gpu_memory():
    """Print detailed information about GPU memory usage."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    print("\nGPU Memory Information:")
    print("-" * 50)

    # Get number of GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

    for i in range(n_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        # Get memory allocated
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert to MB
        # Get memory reserved
        reserved = torch.cuda.memory_reserved(i) / 1024**2  # Convert to MB
        # Get total memory
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # Convert to MB
        # Calculate free memory
        free = total - allocated

        print(f"Total Memory: {total:.2f} MB")
        print(f"Allocated Memory: {allocated:.2f} MB ({allocated/total*100:.1f}%)")
        print(f"Reserved Memory: {reserved:.2f} MB ({reserved/total*100:.1f}%)")
        print(f"Free Memory: {free:.2f} MB ({free/total*100:.1f}%)")

    print("-" * 50)

async def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
>>>>>>> vllm
    local_dir: str = "./saved_model",
    use_chat_api: bool = True,
    prompt: str = "Explain reinforcement learning in a few sentences",
    system_message: str = "You are a helpful AI assistant.",
    max_tokens: int = 256,
<<<<<<< HEAD
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_completions: int = 1,
    use_bfloat16: bool = False,
    verbose: bool = True,
):
    """
    Test script for vLLM - download a model from Hugging Face, save it locally, 
    start a vLLM server with it, and generate some text.
    
=======
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    num_completions: int = 10,
    use_bfloat16: bool = True,
    verbose: bool = True,
    max_model_len: int = 2048,
    extra_generation_args: Optional[Dict[str, Any]] = None,
    vllm_server_args: Optional[Dict[str, Any]] = None,
    extra_parameters: Optional[Dict[str, Any]] = {},
):
    """
    Test script for vLLM - download a model from Hugging Face, save it locally,
    start a vLLM server with it, generate completions, and then kill the server.

>>>>>>> vllm
    Args:
        model_name: Name of the model on Hugging Face to use
        local_dir: Local directory to save the model to
        use_chat_api: Whether to use the chat API or completion API
        prompt: Text prompt for generation
        system_message: System message for chat completions
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p value for sampling
<<<<<<< HEAD
        num_completions: Number of completions to generate
        use_bfloat16: Whether to use bfloat16 precision
        verbose: Whether to print verbose output
    """
    global current_vllm_instance
    vllm_instance = None
    
=======
        top_k: Top-k value for sampling
        num_completions: Number of completions to generate
        use_bfloat16: Whether to use bfloat16 precision
        verbose: Whether to print verbose output
        max_model_len: Maximum model length for vLLM
        extra_generation_args: Additional generation arguments to pass to the vLLM API
        vllm_server_args: Additional server configuration arguments to pass to vLLM
    """
    global current_vllm_instance
    vllm_instance = None

>>>>>>> vllm
    try:
        # Step 1: Download and save the model locally
        if verbose:
            print(f"Downloading model: {model_name}")
<<<<<<< HEAD
        
        # Determine dtype based on argument
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        
        # Download and save the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype
        )
        
=======

        # Determine dtype based on argument
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16

        # Download and save the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

>>>>>>> vllm
        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"  # This is important for vLLM
        )
<<<<<<< HEAD
        
        # If the tokenizer doesn't have a pad token, use the eos token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
=======
        print(f"tokenizer.pad_token: {tokenizer.pad_token}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")

        # If the tokenizer doesn't have a pad token, use the eos token
        if tokenizer.pad_token is None:
            print(f"here?")
            tokenizer.pad_token = tokenizer.eos_token

        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

>>>>>>> vllm
        # Save model and tokenizer
        if verbose:
            print(f"Saving model and tokenizer to: {local_dir}")
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
<<<<<<< HEAD
        
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
        
=======

        # Free memory after saving
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Step 2: Start vLLM server
        if verbose:
            print("Starting vLLM server...")

        # Use the vllm_server_args passed from the main function

        # Create logs directory if it doesn't exist
        os.makedirs("./logs", exist_ok=True)

        # Set default vllm server args if none provided
        server_args = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": max_model_len,
            "disable_log_requests": True,
            "enable_prefix_caching": True,
        }

        # Update with custom args if provided
        if vllm_server_args:
            server_args.update(vllm_server_args)

>>>>>>> vllm
        # Start the vLLM server
        vllm_instance = await start_vllm(
            model=local_dir,
            log_file="./logs/test_vllm.log",
<<<<<<< HEAD
            max_concurrent_requests=8,
            named_arguments=vllm_server_args,
            verbosity=2 if verbose else 1,
            timeout=180.0,  # 3 minutes timeout
        )
        
        # Set the global instance for signal handling
        current_vllm_instance = vllm_instance
        
=======
            max_concurrent_requests=num_completions * 2,
            named_arguments=server_args,
            verbosity=2 if verbose else 1,
            timeout=300.0,  # 5 minutes timeout for server startup
        )

        # Set the global instance for signal handling
        current_vllm_instance = vllm_instance

>>>>>>> vllm
        # Step 3: Generate text
        generation_args = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
<<<<<<< HEAD
        
        if verbose:
            print(f"\nGenerating {'chat' if use_chat_api else 'completion'} with prompt:\n{prompt}\n")
        
=======

        # Merge extra generation arguments if provided
        if extra_generation_args:
            generation_args.update(extra_generation_args)

        if verbose:
            print(f"\nGenerating {'chat' if use_chat_api else 'completion'} with prompt:\n{prompt}\n")
            print(f"Generating {num_completions} completions...")

>>>>>>> vllm
        # Generate text using the appropriate function
        start_time = time.time()
        if use_chat_api:
            completions = await vllm_chat_generate_text(
                vllm_instance=vllm_instance,
                prompts=[prompt],
                num_completions=num_completions,
                generation_args=generation_args,
<<<<<<< HEAD
                system_message=system_message
=======
                system_message=system_message,
                #assistant_messages=["<think>"],
                extra_parameters=extra_parameters,
>>>>>>> vllm
            )
        else:
            completions = await vllm_generate_text(
                vllm_instance=vllm_instance,
                prompts=[prompt],
                num_completions=num_completions,
<<<<<<< HEAD
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
        
=======
                generation_args=generation_args,
                extra_parameters=extra_parameters,
            )

        generation_time = time.time() - start_time

        # Calculate perplexity for each completion using the downloaded model
        if verbose:
            print("\nCalculating perplexity for each completion...")

        # Load the tokenizer (already downloaded)
        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        # Load the model again for perplexity calculation
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            torch_dtype=dtype
        ).eval()

        # If CUDA is available, move model to GPU
        if torch.cuda.is_available():
            model = model.cuda()

        perplexities = []
        for i, completion in enumerate(completions):
            # Calculate perplexity based on log probabilities
            input_ids = tokenizer.encode(completion, return_tensors="pt")

            # Move input to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Print generated completions with perplexity and token count
        print("\n" + "="*50)
        print(f"Generated {len(completions)} completion(s) in {generation_time:.2f}s:")
        for i, (completion, perplexity) in enumerate(zip(completions, perplexities)):
            # Calculate number of tokens in the completion
            tokens = tokenizer.encode(completion)
            token_count = len(tokens)

            print(f"\n--- Completion {i+1}/{num_completions} ---")
            print(f"Perplexity: {perplexity:.4f}")
            print(f"Number of tokens: {token_count}")
            print(completion)
        print("="*50)

>>>>>>> vllm
        # Step 4: Clean up
        if verbose:
            print("\nStopping vLLM server...")
        await stop_vllm(vllm_instance)
        if verbose:
            print("vLLM server stopped successfully")
<<<<<<< HEAD
        
=======

>>>>>>> vllm
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources if needed
<<<<<<< HEAD
        if vllm_instance is not None and vllm_instance.process.returncode is None:
            try:
                await stop_vllm(vllm_instance)
            except Exception:
                pass
        
        # Reset the global instance
        current_vllm_instance = None
        
        # Make sure to clean up any remaining processes
        kill_vllm_workers()
        
=======
        if vllm_instance is not None and vllm_instance.process and vllm_instance.process.returncode is None:
            try:
                await stop_vllm(vllm_instance)
                print("vLLM server stopped in finally block")
            except Exception as e:
                print(f"Error stopping vLLM in finally block: {e}")

        # Reset the global instance
        current_vllm_instance = None

        # Make sure to clean up any remaining processes
        kill_vllm_workers()

        # Double-check that no vLLM processes are left running with multiple attempts
        for _ in range(3):  # Try multiple times to ensure cleanup
            kill_vllm_workers()
            await asyncio.sleep(1)  # Give some time between kill attempts

>>>>>>> vllm
        if verbose:
            print("All cleanup complete")

if __name__ == "__main__":
<<<<<<< HEAD
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
=======
    # Define variables here using the same defaults as in train_branch_vllm.py
    model = "Qwen/Qwen2.5-3B-Instruct"
    local_dir = "./saved_model"
    prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
Using a list of numbers, find a mathematical expression that equals the target.
You can use addition (+), subtraction (-) and multiplication (*). Do not use division (/).
You must use each number at most once, and you don't have to use all numbers.
The numbers cannot be concatenated (e.g., you cannot use 9 and 5 to make 95).
Write just the left side of the expression.

Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.


Think about the solution and write it step by step.

Task:

Numbers: [5, 8]
Target: 13
"""
    #prompt = "What is the capital of France?, answer briefly"
    system_message = "You are a helpful AI assistant."
    max_tokens = 512
    temperature = 1.0
    top_p = 1.0
    top_k = 50
    num_completions = 5  # Generate 10 completions as requested
    use_completion = False  # False means use chat API
    use_bfloat16 = True
    quiet = False
    max_model_len = 2048

    # Example of extra generation args - add additional arguments here
    extra_args = {
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop": ["<|endoftext|>"]
        #"seed": 42,
    }
    extra_parameters = {
        "ignore_eos": False,
        "skip_special_tokens": False,
    }

    # vLLM server arguments
    vllm_server_args = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": max_model_len,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        #"enforce_eager": False,
        "generation_config": "vllm",
    }

    try:
        # Run the async main function with defined variables
        asyncio.run(main(
            model_name=model,
            local_dir=local_dir,
            use_chat_api=not use_completion,
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_completions=num_completions,
            use_bfloat16=use_bfloat16,
            verbose=not quiet,
            max_model_len=max_model_len,
            extra_generation_args=extra_args,
            vllm_server_args=vllm_server_args,
            extra_parameters=extra_parameters,
>>>>>>> vllm
        ))
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    finally:
        # Ensure all vLLM processes are cleaned up
        kill_vllm_workers()
<<<<<<< HEAD
=======

        # Emergency cleanup: Try multiple times to ensure all processes are killed
        print("Emergency cleanup: Ensuring all vLLM processes are terminated...")
        # Try multiple times to ensure cleanup
        for i in range(5):
            kill_vllm_workers()
            time.sleep(1)  # Give some time between kill attempts

>>>>>>> vllm
        print("Final cleanup complete")