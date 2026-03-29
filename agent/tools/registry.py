import json
from typing import Dict, List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from agent.tools.base import ToolContext
from agent.tools.base import BaseTool
from agent.tools.grounding import (
    QueryParseTool, ReadImageFilesTool, MasksFilterTool, VlmFilterTool,
    VlmScoreTool, ArgmaxImageAndSegIdTool, SegmentTargetInReferenceTool,
    VlmFrameExpansionTool, ExpandFromSecondaryViewTool, 
    SegmentAllTargetObjectTool, ReconstructPointCloudTool, CalculateBboxTool, CentroidCompleteTool
)

class ToolRegistry:
    def __init__(self, context: ToolContext):
        self.context = context
        self.tools: Dict[str, BaseTool] = {}
        self._register_all()
        print("Registered tools: ", self.tools.keys())

    def _register_all(self):
        tool_classes = [
            QueryParseTool, ReadImageFilesTool, MasksFilterTool, VlmFilterTool,
            VlmScoreTool, ArgmaxImageAndSegIdTool, SegmentTargetInReferenceTool,
            VlmFrameExpansionTool, ExpandFromSecondaryViewTool, 
            SegmentAllTargetObjectTool, ReconstructPointCloudTool, CentroidCompleteTool, CalculateBboxTool
        ]
        
        for cls in tool_classes:
            tool = cls(self.context)
            self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        return self.tools.get(name)

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        lines = []
        for tool in self.tools.values():
            schema = tool.schema['function']
            lines.append(f"## Tool: {schema['name']}")
            lines.append(f"Description: {schema['description']}")
            lines.append(f"Parameters: {json.dumps(schema['parameters'], indent=2)}")
            if tool.return_description:
                lines.append(f"Returns: {tool.return_description}")
            lines.append("-" * 30)
        return "\n".join(lines)