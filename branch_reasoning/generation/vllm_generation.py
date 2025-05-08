import asyncio
import time
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
import os
import socket
import subprocess
import sys
import re
import torch
from typing import Any, IO, Optional, List, Dict, Union, Callable

@dataclass
class vLLM:
    client: AsyncOpenAI
    max_concurrent_tokens: int
    model: str
    process: asyncio.subprocess.Process


def kill_vllm_workers() -> None:
    """
    Kill all vLLM worker processes that might be running.
    This helps ensure a clean start for new vLLM server instances.
    """
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    pids = [
        line.split()[1]
        for line in result.stdout.splitlines()
        if "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=" in line
    ]
    for pid in pids:
        try:
            subprocess.run(["kill", "-9", pid], check=False)
        except Exception:
            # Ignore errors if process cannot be killed
            pass


async def start_vllm(
    model: str,
    env: Optional[Dict[str, str]] = None,
    log_file: str = "./logs/vllm.log",
    max_concurrent_requests: int = 128,
    named_arguments: Dict[str, Any] = {},
    timeout: float = 120.0,
    verbosity: int = 2,
) -> vLLM:
    """
    Start a vLLM server as an async subprocess and return a vLLM client.

    Args:
        model: The name or path of the model to serve.
        env: Environment variables to set for the vLLM process.
        log_file: Path to write logs to.
        max_concurrent_requests: Maximum number of concurrent requests.
        named_arguments: Additional arguments to pass to the vLLM server.
        timeout: Maximum time to wait for the server to start.
        verbosity: How verbose the output should be (0=silent, 1=minimal, 2=detailed).

    Returns:
        A vLLM object with a client connected to the server.
    """
    # Ensure we don't have stray vLLM processes
    t0 = time.time()
    kill_vllm_workers()
    
    # Handle local model paths
    if os.path.exists(os.path.abspath(model)):
        named_arguments.setdefault("served_model_name", model)
        model = os.path.abspath(model)
    
    # Find an available port
    port = named_arguments.get("port") or 8000
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((named_arguments.get("host") or "0.0.0.0", port))
            break
        except socket.error:
            if "port" in named_arguments and named_arguments["port"] == port:
                raise RuntimeError(f"Port {port} is already in use")
            port += 1
        finally:
            sock.close()
    
    named_arguments["port"] = port
    
    # Build the command line arguments
    args = [
        "vllm",
        "serve",
        model,
        *[
            f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
            for key, value in named_arguments.items()
        ],
        "--api-key=default",
    ]
    
    # Start the vLLM process
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            **(env or {}),
        },
    )
    
    if verbosity > 0:
        print(f"$ {' '.join(args)}")
    
    # Set up logging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log = open(log_file, "w")
    logging = verbosity > 1
    max_concurrent_tokens: Optional[int] = None

    async def log_output(stream: asyncio.StreamReader, io: IO[str]) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded_line = line.decode()
            if logging:
                io.write(decoded_line)
                io.flush()
            log.write(decoded_line)
            log.flush()
            nonlocal max_concurrent_tokens
            if not max_concurrent_tokens:
                match = re.search(
                    r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x",
                    decoded_line,
                )
                if match:
                    max_concurrent_tokens = int(
                        int(match.group(1)) * float(match.group(2))
                    )
        log.close()

    if process.stdout:
        asyncio.create_task(log_output(process.stdout, sys.stdout))
    if process.stderr:
        asyncio.create_task(log_output(process.stderr, sys.stderr))
    
    # Create the OpenAI-compatible client
    client = AsyncOpenAI(
        api_key="default",
        base_url=f"http://{named_arguments.get('host', '0.0.0.0')}:{named_arguments['port']}/v1",
        max_retries=6,
        http_client=DefaultAsyncHttpxClient(
            limits=httpx.Limits(
                max_connections=max_concurrent_requests,
                max_keepalive_connections=max_concurrent_requests,
            ),
            timeout=httpx.Timeout(timeout=1_200, connect=10.0),
        ),
    )
    
    # Wait for server to become responsive
    start = asyncio.get_event_loop().time()
    sleep_time = 1
    while True:
        try:
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=named_arguments.get("served_model_name", model),
                max_tokens=1,
            )
            break
        except Exception:
            if asyncio.get_event_loop().time() - start > timeout:
                process.terminate()
                kill_vllm_workers()
                raise TimeoutError("vLLM server did not start in time")
            await asyncio.sleep(sleep_time)  # Add a small delay to prevent tight loops
            sleep_time *= 1.1
            continue
    
    if logging:
        print(f"vLLM server started successfully. Logs can be found at {log_file}")
        logging = False
    
    if max_concurrent_tokens is None and False:
        print(f"In the last in")
        process.terminate()
        kill_vllm_workers()
        raise RuntimeError(
            "Max concurrent requests for the maximum model length not logged"
        )
    
    return vLLM(
        client,
        max_concurrent_tokens,
        named_arguments.get("served_model_name", model),
        process,
    )


async def vllm_generate_text(
    vllm_instance: vLLM,
    prompts: List[str],
    num_completions: int = 1,
    max_seq_len: int = 2048,
    generation_args: Dict[str, Any] = {},
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate text using vLLM, similar to the HuggingFace generate_text function.
    
    Args:
        vllm_instance: A running vLLM instance
        prompts: List of prompts to generate completions for
        num_completions: Number of completions to generate per prompt
        max_seq_len: Maximum sequence length for generation
        generation_args: Additional generation parameters
        
    Returns:
        List of generated text completions
    """
    # Set up generation parameters
    generation_params = {
        "max_tokens": generation_args.get("max_new_tokens", 512),
    }
    
    # Map parameters to OpenAI API format
    generation_params["temperature"] = temperature
    if "top_p" in generation_args:
        generation_params["top_p"] = generation_args["top_p"]
    
    # Note: OpenAI API doesn't support top_k directly, we omit it for compatibility
    # See OpenAI API docs: https://platform.openai.com/docs/api-reference/completions
    
    # For OpenAI API compatibility
    generation_params["n"] = num_completions
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(min(16, len(prompts)))
    
    async def generate_completion(prompt: str) -> List[str]:
        async with semaphore:
            try:
                response = await vllm_instance.client.completions.create(
                    model=vllm_instance.model,
                    prompt=prompt,
                    **generation_params
                )
                return [(choice.text, prompt) for choice in response.choices]
            except Exception as e:
                print(f"Error generating completion: {e}")
                if "top_k" in str(e) and "top_k" in generation_args:
                    print("Retrying without top_k parameter...")
                    # If the error is related to top_k, retry without it
                    retry_params = generation_params.copy()
                    if "top_k" in retry_params:
                        del retry_params["top_k"]
                    
                    response = await vllm_instance.client.completions.create(
                        model=vllm_instance.model,
                        prompt=prompt,
                        **retry_params
                    )
                    return [(choice.text, prompt) for choice in response.choices]
                raise
    
    # Gather all completion tasks
    all_generated_texts = []
    tasks = [generate_completion(prompt) for prompt in prompts]
    
    for completed_task in await asyncio.gather(*tasks):
        all_generated_texts.extend([prompt + completion for completion, prompt in completed_task])
    
    return all_generated_texts


async def vllm_chat_generate_text(
    vllm_instance: vLLM,
    prompts: List[str],
    num_completions: int = 1,
    max_seq_len: int = 2048,
    generation_args: Dict[str, Any] = {},
    system_message: str = "You are a helpful assistant.",
) -> List[str]:
    """
    Generate text using vLLM's chat completions API, which is more appropriate
    for modern LLMs that are trained with a chat format.
    
    Args:
        vllm_instance: A running vLLM instance
        prompts: List of prompts to generate completions for
        num_completions: Number of completions to generate per prompt
        max_seq_len: Maximum sequence length for generation
        generation_args: Additional generation parameters
        system_message: System message to use for each chat completion
        
    Returns:
        List of generated text completions
    """
    # Set up generation parameters
    generation_params = {
        "max_tokens": generation_args.get("max_new_tokens", 512),
    }
    
    # Map parameters to OpenAI API format
    if "temperature" in generation_args:
        generation_params["temperature"] = generation_args["temperature"]
    if "top_p" in generation_args:
        generation_params["top_p"] = generation_args["top_p"]
    
    # Note: OpenAI API doesn't support top_k directly for chat completions
    # See OpenAI API docs: https://platform.openai.com/docs/api-reference/chat/create
    
    # For OpenAI API compatibility
    generation_params["n"] = num_completions
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(min(16, len(prompts)))
    
    async def generate_chat_completion(prompt: str) -> List[str]:
        async with semaphore:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = await vllm_instance.client.chat.completions.create(
                    model=vllm_instance.model,
                    messages=messages,
                    **generation_params
                )
                return [choice.message.content for choice in response.choices]
            except Exception as e:
                print(f"Error generating chat completion: {e}")
                if "top_k" in str(e) and "top_k" in generation_args:
                    print("Retrying without top_k parameter...")
                    # If the error is related to top_k, retry without it
                    retry_params = generation_params.copy()
                    if "top_k" in retry_params:
                        del retry_params["top_k"]
                    
                    response = await vllm_instance.client.chat.completions.create(
                        model=vllm_instance.model,
                        messages=messages,
                        **retry_params
                    )
                    return [choice.message.content for choice in response.choices]
                raise
    
    # Gather all completion tasks
    all_generated_texts = []
    tasks = [generate_chat_completion(prompt) for prompt in prompts]
    
    for completed_task in await asyncio.gather(*tasks):
        all_generated_texts.extend(completed_task)
    
    return all_generated_texts


async def stop_vllm(vllm_instance: vLLM) -> None:
    """
    Cleanly stop a running vLLM instance and clean up resources.
    
    Args:
        vllm_instance: A running vLLM instance
    """
    if vllm_instance.process.returncode is None:
        try:
            vllm_instance.process.terminate()
            await vllm_instance.process.wait()
        except Exception:
            pass
    
    # Make sure all worker processes are terminated
    kill_vllm_workers()