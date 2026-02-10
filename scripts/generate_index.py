import os
import json
import re

import yaml

def parse_frontmatter(content):
    """
    Parses YAML frontmatter using PyYAML for standard compliance.
    """
    fm_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        return {}
    
    try:
        return yaml.safe_load(fm_match.group(1)) or {}
    except yaml.YAMLError as e:
        print(f"⚠️ YAML parsing error: {e}")
        return {}

def generate_index(skills_dir, output_file):
    print(f"🏗️ Generating index from: {skills_dir}")
    skills = []

    for root, dirs, files in os.walk(skills_dir):
        # Skip .disabled or hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        if "SKILL.md" in files:
            skill_path = os.path.join(root, "SKILL.md")
            dir_name = os.path.basename(root)
            parent_dir = os.path.basename(os.path.dirname(root))
            
            # Default values
            skill_info = {
                "id": dir_name,
                "path": os.path.relpath(root, os.path.dirname(skills_dir)),
                "category": parent_dir if parent_dir != "skills" else "uncategorized",
                "name": dir_name.replace("-", " ").title(),
                "description": "",
                "risk": "unknown",
                "source": "unknown"
            }
            
            try:
                with open(skill_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Error reading {skill_path}: {e}")
                continue

            # Parse Metadata
            metadata = parse_frontmatter(content)
            
            # Merge Metadata
            if "name" in metadata: skill_info["name"] = metadata["name"]
            if "description" in metadata: skill_info["description"] = metadata["description"]
            if "risk" in metadata: skill_info["risk"] = metadata["risk"]
            if "source" in metadata: skill_info["source"] = metadata["source"]
            
            # Fallback for description if missing in frontmatter (legacy support)
            if not skill_info["description"]:
                body = content
                fm_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
                if fm_match:
                    body = content[fm_match.end():].strip()
                
                # Simple extraction of first non-header paragraph
                lines = body.split('\n')
                desc_lines = []
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        if desc_lines: break
                        continue
                    desc_lines.append(line.strip())
                
                if desc_lines:
                    skill_info["description"] = " ".join(desc_lines)[:250].strip()

            skills.append(skill_info)

    # Sort validation: by name
    skills.sort(key=lambda x: (x["name"].lower(), x["id"].lower()))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(skills, f, indent=2)
    
    print(f"✅ Generated rich index with {len(skills)} skills at: {output_file}")
    return skills

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skills_path = os.path.join(base_dir, "skills")
    output_path = os.path.join(base_dir, "skills_index.json")
    generate_index(skills_path, output_path)
