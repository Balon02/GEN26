import os
import kagglehub
from dotenv import load_dotenv

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

from gemma import gm

load_dotenv('/home/balon/source/GEN26/.env')
print(os.environ['KAGGLE_USERNAME'])

GEMMA_PATH = kagglehub.model_download("google/gemma-4/flax/gemma-4-e2b-it")
# CKPT_PATH = os.path.join(GEMMA_PATH, "gemma-4-e2b-it")
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')


def main():
    print('INIT')
    model = gm.nn.Gemma4_E2B(text_only=False)
    print('MODEL_LOAD')
    params = gm.ckpts.load_params(GEMMA_PATH, quantize=True)
    print('TOKENIZER_LOAD')
    tokenizer = gm.text.Gemma4Tokenizer(TOKENIZER_PATH)

    print('CHAT_SETUP')
    chatbot = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
    tokenizer=tokenizer,
    )

    stop = False
    while not stop:
        text = input("enter prompt, 'q' to quit:\n")
        if text not in ['q', 'quit', 'exit']:
            print(chatbot.chat(text))
        else: 
            stop = True


if __name__ == '__main__':
    main()
