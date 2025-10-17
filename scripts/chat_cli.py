"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli -i mid
"""
import argparse
import contextlib
import torch
from nanochat.common import compute_init
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument(
    '-d', '--device',
    type=str,
    default='auto',
    choices=['auto', 'cuda', 'cpu'],
    help="Device to run inference on (default: auto-detect CUDA, else CPU)"
)
args = parser.parse_args()

# Init the model and tokenizer
if args.device == 'auto':
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    target_device = args.device

autocast_ctx = contextlib.nullcontext()
if target_device == 'cuda':
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    if args.device == 'cuda':
        raise RuntimeError("CUDA was requested explicitly but is not available.")
    ddp = False
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    device = torch.device("cpu")
    torch.manual_seed(42)
    print("Running chat_cli on CPU. Generation will be significantly slower.")

try:
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
except torch.cuda.OutOfMemoryError as oom:
    if target_device == 'cuda' and args.device == 'auto':
        print("CUDA out of memory while loading the model. Falling back to CPU. Generation will be significantly slower.")
        torch.cuda.empty_cache()
        target_device = 'cpu'
        ddp = False
        ddp_rank = ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device("cpu")
        torch.manual_seed(42)
        autocast_ctx = contextlib.nullcontext()
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    else:
        raise oom

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
