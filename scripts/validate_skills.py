import os
import re
import argparse
import sys

WHEN_TO_USE_PATTERNS = [
    re.compile(r"^##\s+When\s+to\s+Use", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^##\s+Use\s+this\s+skill\s+when", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^##\s+When\s+to\s+Use\s+This\s+Skill", re.MULTILINE | re.IGNORECASE),
]

def has_when_to_use_section(content):
    return any(pattern.search(content) for pattern in WHEN_TO_USE_PATTERNS)

def parse_frontmatter(content):
    """
    Simple frontmatter parser using regex to avoid external dependencies.
    Returns a dict of key-values.
    """
    fm_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        return None
    
    fm_text = fm_match.group(1)
    metadata = {}
    for line in fm_text.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata

def validate_skills(skills_dir, strict_mode=False):
    print(f"🔍 Validating skills in: {skills_dir}")
    print(f"⚙️  Mode: {'STRICT (CI)' if strict_mode else 'Standard (Dev)'}")
    
    errors = []
    warnings = []
    skill_count = 0
    
    # Pre-compiled regex
    security_disclaimer_pattern = re.compile(r"AUTHORIZED USE ONLY", re.IGNORECASE)

    valid_risk_levels = ["none", "safe", "critical", "offensive"]

    for root, dirs, files in os.walk(skills_dir):
        # Skip .disabled or hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        if "SKILL.md" in files:
            skill_count += 1
            skill_path = os.path.join(root, "SKILL.md")
            rel_path = os.path.relpath(skill_path, skills_dir)
            
            try:
                with open(skill_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                errors.append(f"❌ {rel_path}: Unreadable file - {str(e)}")
                continue
            
            # 1. Frontmatter Check
            metadata = parse_frontmatter(content)
            if not metadata:
                errors.append(f"❌ {rel_path}: Missing or malformed YAML frontmatter")
                continue # Cannot proceed without metadata

            # 2. Metadata Schema Checks
            if "name" not in metadata:
                errors.append(f"❌ {rel_path}: Missing 'name' in frontmatter")
            elif metadata["name"] != os.path.basename(root):
                warnings.append(f"⚠️  {rel_path}: Name '{metadata['name']}' does not match folder name '{os.path.basename(root)}'")

            if "description" not in metadata:
                errors.append(f"❌ {rel_path}: Missing 'description' in frontmatter")

            # Risk Validation (Quality Bar)
            if "risk" not in metadata:
                msg = f"⚠️  {rel_path}: Missing 'risk' label (defaulting to 'unknown')"
                if strict_mode: errors.append(msg.replace("⚠️", "❌"))
                else: warnings.append(msg)
            elif metadata["risk"] not in valid_risk_levels:
                errors.append(f"❌ {rel_path}: Invalid risk level '{metadata['risk']}'. Must be one of {valid_risk_levels}")

            # Source Validation
            if "source" not in metadata:
                msg = f"⚠️  {rel_path}: Missing 'source' attribution"
                if strict_mode: errors.append(msg.replace("⚠️", "❌"))
                else: warnings.append(msg)

            # 3. Content Checks (Triggers)
            if not has_when_to_use_section(content):
                msg = f"⚠️  {rel_path}: Missing '## When to Use' section"
                if strict_mode: errors.append(msg.replace("⚠️", "❌"))
                else: warnings.append(msg)

            # 4. Security Guardrails
            if metadata.get("risk") == "offensive":
                if not security_disclaimer_pattern.search(content):
                    errors.append(f"🚨 {rel_path}: OFFENSIVE SKILL MISSING SECURITY DISCLAIMER! (Must contain 'AUTHORIZED USE ONLY')")

    # Reporting
    print(f"\n📊 Checked {skill_count} skills.")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} Warnings:")
        for w in warnings:
            print(w)

    if errors:
        print(f"\n❌ Found {len(errors)} Critical Errors:")
        for e in errors:
            print(e)
        return False

    if strict_mode and warnings:
        print("\n❌ STRICT MODE: Failed due to warnings.")
        return False

    print("\n✨ All skills passed validation!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Antigravity Skills")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings (for CI)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skills_path = os.path.join(base_dir, "skills")
    
    success = validate_skills(skills_path, strict_mode=args.strict)
    if not success:
        sys.exit(1)
