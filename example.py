from llama_cpp.llama import Llama, LlamaGrammar
import json

system_prompt = open("system_prompt.txt", "r").read()

grammar_json = open("json.gbnf", "r").read()
grammar_json_arr = open("json_arr.gbnf", "r").read()
grammar = LlamaGrammar.from_string(grammar_json)

models = {
  "llama3": {
    "chat_format": "llama-3",
    "file": "/models/lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    "context_length": 8192
  },
  "phi3": {
    # "chat_format": "phi-3",
    "chat_format": "chatml",
    "file": "/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
    "context_length": 4096
  }
}

model = models["phi3"]

llm = Llama(model["file"], n_gpu_layers=-1, chat_format=model["chat_format"], n_ctx=model["context_length"])

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": "John realizes he's living in a simulation"},
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
              "affected_attribute": {"type": "string"},
              "amount": {"type": "number"},
              "mood": {"type": "string"},
              "event_description": {"type": "string"},
              "inner_thoughts": {"type": "string"},
            },
            "required": ["affected_attribute", "amount", "mood", "event_description", "inner_thoughts"],
        },
    },
    temperature=0.7,
)

print(json.dumps(json.loads(response['choices'][0]['message']['content']), indent=4))
