from typing import List, Dict

class Memory:
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []

    def add_system_message(self, message: str):
        self.chat_history.append({"role": "system", "content": [{"type": "text", "text": message},],})

    def add_user_message(self, message: str):
        self.chat_history.append({"role": "user", "content": [{"type": "text", "text": message},],})

    def add_assistant_message(self, message: str):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": message},],})

    def get_chat_history(self,):
        return self.chat_history

    def clear(self):
        self.chat_history = []