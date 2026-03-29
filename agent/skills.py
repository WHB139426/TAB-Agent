import os
import glob
import yaml

class Skill:
    def __init__(self, name, description, content):
        self.name = name
        self.description = description
        self.content = content

class SkillLibrary:
    def __init__(self, skill_dir="./agent/skills"):
        self.skill_dir = skill_dir
        self.skills = {}

    def load_skills(self):
        search_path = os.path.join(self.skill_dir, "**", "SKILL.md")
        skill_files = glob.glob(search_path, recursive=True)
        
        print(f"Found {len(skill_files)} skills in {self.skill_dir}")

        for file_path in skill_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter_str = parts[1]
                        content = parts[2].strip()
                        
                        meta = yaml.safe_load(frontmatter_str)
                        name = meta.get('name', 'unknown')
                        description = meta.get('description', '')
                        
                        self.skills[name] = Skill(name, description, content)
                        print(f"Loaded skill: {name}")
                else:
                    print(f"Skipping {file_path}: No frontmatter found.")
            except Exception as e:
                print(f"Error loading skill {file_path}: {e}")

    def get_skill_registry_text(self):
        """
        - visual_grounding: Finds 3D objects in video...
        - email_sender: Sends emails...
        """
        lines = []
        for name, skill in self.skills.items():
            lines.append(f"- {name}: {skill.description}")
        return "\n".join(lines)

    def has_skill(self, skill_name):
        return skill_name in self.skills

    def get_skill_description(self, skill_name):
        if skill_name in self.skills:
            return self.skills[skill_name].content
        return ""

    def get_all_skill_descriptions(self):
        return "\n\n".join([s.content for s in self.skills.values()])