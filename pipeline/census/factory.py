# pipeline/census/factory.py
import redis
from pipeline.census.census_manager import CensusManager

def get_default_manager(total_beds: int = 20) -> CensusManager:
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return CensusManager(client, total_beds=total_beds)