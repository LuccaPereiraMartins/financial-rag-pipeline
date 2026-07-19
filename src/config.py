import os

from dotenv import load_dotenv
load_dotenv()

# don't set all of these as env variables because they shouldn't be easily changed
class Config:
    OPENAI_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    INDEX_DIR = "data/index"
    # in practice, these would be tuned for the specific data against some evaluation framework
    CHUNK_SIZE = 500 #NOTE: small chunk size because each line can be dense with information
    CHUNK_OVERLAP = 50
    RETRIEVAL_TOP_K = 8
    MAX_AGENT_TOOL_CALLS = 8
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COLLECTION_NAME = "earnings_chunks"

# fail early if the API key is not set. no need to handle elsewhere in the code
assert Config.OPENAI_API_KEY, "OPENAI_API_KEY is not set"