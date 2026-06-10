import os
import cv2
import argparse
import kagglehub
from dotenv import load_dotenv

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="vmm"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

from gemma import gm, peft
from jax import numpy as jnp

load_dotenv('/home/balon/source/GEN26/.env')
print(os.environ['KAGGLE_USERNAME'])

GEMMA_PATH = kagglehub.model_download("google/gemma-3/flax/gemma3-4b-it")
CKPT_PATH = os.path.join(GEMMA_PATH, "gemma3-4b-it")
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')


def main():
    print('INIT')
    model = model=gm.nn.Gemma3_4B(text_only=False)
    print('MODEL_LOAD')
    params = gm.ckpts.load_params(CKPT_PATH)
    print('TOKENIZER_LOAD')
    tokenizer = gm.text.Gemma3Tokenizer(TOKENIZER_PATH)

    print('CHAT_SETUP')
    chatbot = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
    tokenizer=tokenizer,
    print_stream=True
    )

    stop = False
    while not stop:
        text = input("enter prompt, 'q' to quit:\n")
        if text not in ['q', 'quit', 'exit']:
            # image = cv2.resize(cv2.cvtColor(cv2.imread('/home/balon/source/ML_HUB_2025/runs/pcb2pcb_classifier/cvit_results.png'), cv2.COLOR_BGR2RGB), (896, 896), interpolation=cv2.INTER_CUBIC)
            # chatbot.chat(f'{text} <|image|>', images=[image])
            chatbot.chat(text)
        else: 
            stop = True


if __name__ == '__main__':
    main()
