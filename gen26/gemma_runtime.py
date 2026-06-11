from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


# Match smoke_test_g3.py before importing Gemma/JAX.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"

import kagglehub  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gemma import gm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
GEMMA_MODEL = "google/gemma-3/flax/gemma3-4b-it"


class GemmaDigestRuntime:
    def __init__(self) -> None:
        load_dotenv(str(REPO_ROOT / ".env"))

        print("INIT", flush=True)
        gemma_path = kagglehub.model_download(GEMMA_MODEL)
        ckpt_path = os.path.join(gemma_path, "gemma3-4b-it")
        tokenizer_path = os.path.join(gemma_path, "tokenizer.model")

        self.model = gm.nn.Gemma3_4B(text_only=False)
        print("MODEL_LOAD", flush=True)
        self.params = gm.ckpts.load_params(ckpt_path)
        print("TOKENIZER_LOAD", flush=True)
        self.tokenizer = gm.text.Gemma3Tokenizer(tokenizer_path)
        self.cache_length = 10240
        self.safe_input_tokens = 7800
        self.max_output_tokens = 768
        self.image_height = self.model.config.vision_encoder.image_height
        self.image_width = self.model.config.vision_encoder.image_width
        if self.image_height != self.image_width:
            raise ValueError("Expected square Gemma image input.")
        self.image_size = self.image_height

        print("CHAT_SETUP", flush=True)
        self._new_chatbot()

    def _new_chatbot(self):
        return gm.text.ChatSampler(
            model=self.model,
            params=self.params,
            multi_turn=True,
            tokenizer=self.tokenizer,
            print_stream=True,
            cache_length=self.cache_length,
            max_out_length=self.max_output_tokens,
        )

    def count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_bos=True))

    def chat(
        self,
        prompt: str,
        images: list[object] | None = None,
        stream_file=None,
    ) -> str:
        chatbot = self._new_chatbot()
        stream_capture = StringIO()
        files = [sys.stdout, stream_capture]
        if stream_file is not None:
            files.append(stream_file)
        with redirect_stdout(Tee(*files)):
            if images:
                response = chatbot.chat(
                    prompt,
                    images=images,
                    max_new_tokens=self.max_output_tokens,
                )
            else:
                response = chatbot.chat(prompt, max_new_tokens=self.max_output_tokens)
        if response is None:
            return stream_capture.getvalue().strip()
        if hasattr(response, "text"):
            return response.text
        return str(response)


class Tee:
    def __init__(self, *files) -> None:
        self.files = files

    def write(self, text: str) -> int:
        for file in self.files:
            file.write(text)
        return len(text)

    def flush(self) -> None:
        for file in self.files:
            file.flush()
