from abc import ABC, abstractmethod
from typing import Type, Dict, Any
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolContext:
    client: Any  # LLM Client
    sam_processor: Any
    sam_model: Any
    cache_dir: str
    scannet_video_path: str
    scannet_info_path: str
    
class BaseTool(ABC):
    name: str = ""
    description: str = ""
    return_description: str = ""  
    args_schema: Type[BaseModel] = None 

    def __init__(self, context: ToolContext):
        self.context = context

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass

    @property
    def schema(self) -> Dict[str, Any]:
        if self.args_schema:
            parameters = self.args_schema.model_json_schema()
            parameters.pop("title", None)
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters,
                }
            }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            }
        }