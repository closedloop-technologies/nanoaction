# Chat with the pre-trained model

## From [speedrun.sh](speedrun.sh)

```bash

# Download and load the checkpointed model
python scripts.hf_pull_checkpoint.py

# chat with the model over CLI! Leave out the -p to chat interactively
python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
python -m scripts.chat_web
```

Above will not work until we pull the checkpoints from hugging face.
