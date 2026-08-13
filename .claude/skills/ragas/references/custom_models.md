# Wiring RAGAS to a self-hosted OpenAI-compatible endpoint

RAGAS never talks to a model provider directly — every LLM/embeddings argument (`TestsetGenerator(llm=..., embedding_model=...)`, `evaluate(llm=...)`) must be a `BaseRagasLLM` / `BaseRagasEmbeddings`. The standard way to get one is to build a LangChain LLM/embeddings object and wrap it.

## Pattern

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

custom_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",  # your self-hosted chat-completions endpoint
    api_key="not-needed",                 # SDK requires a non-empty string even when the server ignores it
    model="your-model-name",
)

custom_embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8000/v1",  # your self-hosted embeddings endpoint
    api_key="not-needed",
    model="your-embedding-model-name",
)

llm = LangchainLLMWrapper(custom_llm)
embeddings = LangchainEmbeddingsWrapper(custom_embeddings)

generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
```

RAGAS documents that it "supports all the LLMs and Embeddings available in LangChain" — if you pass a raw LangChain LLM/embeddings object directly to `TestsetGenerator.from_langchain(...)`, RAGAS wraps it for you; the manual `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` calls above are equivalent and needed if you're constructing `TestsetGenerator(...)` directly rather than via the `from_langchain` factory.

## This project's existing pattern

[`clients/embedding_client.py`](../../../clients/embedding_client.py) already does the "self-hosted OpenAI-compatible server, no API key" dance for the raw `openai` SDK:

```python
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
client = OpenAI(base_url=os.environ["EMBEDDING_BASE_URL"], api_key="not-needed")
```

To reuse the same server/model for RAGAS embeddings, wrap it as `OpenAIEmbeddings` (LangChain's client, not the raw `openai` SDK) pointed at the same `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL`, then `LangchainEmbeddingsWrapper(...)` it — see the pattern above.

There is currently **no equivalent chat-completions LLM client** in `clients/` (only `/v1/embeddings` is wired up). RAGAS's testset generation (transforms, persona/query synthesis) and most evaluation metrics need a chat LLM, not just embeddings — that gap needs to be closed (new env vars + a `ChatOpenAI`/`LangchainLLMWrapper` setup, mirroring `embedding_client.py`'s shape) before the `eval/` pipeline can run end-to-end.
