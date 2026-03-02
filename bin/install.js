#!/usr/bin/env node

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');


const REPO = 'https://github.com/legendaryabhi/agent-skills-hub.git';
const HOME = process.env.HOME || process.env.USERPROFILE || '';

function resolveDir(p) {
  if (!p) return null;
  const s = p.replace(/^~($|\/)/, HOME + '$1');
  return path.resolve(s);
}

function findToolReferences(skillDir) {
  const tools = new Set();
  const searchForTools = (dir) => {
    if (!fs.existsSync(dir)) return;
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      if (stat.isDirectory()) {
        searchForTools(fullPath);
      } else if (file.endsWith('.md')) {
        const content = fs.readFileSync(fullPath, 'utf8');
        const regex = /tools\/([a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+)/g;
        let match;
        while ((match = regex.exec(content)) !== null) {
          tools.add(match[1]);
        }
      }
    }
  };
  searchForTools(skillDir);
  return Array.from(tools);
}

function parseArgs() {
  const a = process.argv.slice(2);
  let pathArg = null;
  let versionArg = null;
  let tagArg = null;
  let skillName = null;
  let cursor = false, claude = false, gemini = false, codex = false, openclaw = false;
  let kiro = false, antigravity = false, opencode = false, adal = false;

  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--help' || a[i] === '-h') return { help: true };
    if (a[i] === '--path' && a[i + 1]) { pathArg = a[++i]; continue; }
    if (a[i] === '--version' && a[i + 1]) { versionArg = a[++i]; continue; }
    if (a[i] === '--tag' && a[i + 1]) { tagArg = a[++i]; continue; }
    if (a[i] === '--skill' && a[i + 1]) { skillName = a[++i]; continue; }
    if (a[i] === '--cursor') { cursor = true; continue; }
    if (a[i] === '--claude') { claude = true; continue; }
    if (a[i] === '--gemini') { gemini = true; continue; }
    if (a[i] === '--codex') { codex = true; continue; }
    if (a[i] === '--openclaw') { openclaw = true; continue; }
    if (a[i] === '--kiro') { kiro = true; continue; }
    if (a[i] === '--antigravity') { antigravity = true; continue; }
    if (a[i] === '--opencode') { opencode = true; continue; }
    if (a[i] === '--adal') { adal = true; continue; }
    if (a[i] === 'install') continue;
    if (!a[i].startsWith('-') && !skillName) { skillName = a[i]; continue; }
  }

  return { pathArg, versionArg, tagArg, skillName, cursor, claude, gemini, codex, openclaw, kiro, antigravity, opencode, adal };
}

function defaultDir(opts) {
  if (opts.pathArg) return resolveDir(opts.pathArg);
  if (opts.cursor) return path.join(HOME, '.cursor', 'skills');
  if (opts.claude) return path.join(HOME, '.claude', 'skills');
  if (opts.gemini) return path.join(HOME, '.gemini', 'skills');
  if (opts.codex) {
    const codexHome = process.env.CODEX_HOME;
    if (codexHome) return path.join(codexHome, 'skills');
    return path.join(HOME, '.codex', 'skills');
  }
  if (opts.openclaw) return path.join(HOME, '.openclaw', 'skills');
  if (opts.kiro) return path.join(HOME, '.kiro', 'skills');
  if (opts.antigravity) return path.join(HOME, '.gemini', 'antigravity', 'skills');
  if (opts.opencode) return path.join(HOME, '.agents', 'skills');
  if (opts.adal) return path.join(HOME, '.adal', 'skills');
  return path.join(HOME, '.agent', 'skills');
}

function printHelp() {
  console.log(`
agent-skills-hub — installer

  npx agent-skills-hub [install] [skill-name] [options]

  Clones the skills repo or installs a specific skill.

Options:
  --cursor       Install to ~/.cursor/skills (Cursor IDE)
  --claude       Install to ~/.claude/skills (Claude Code CLI)
  --gemini       Install to ~/.gemini/skills (Gemini CLI)
  --codex        Install to ~/.codex/skills (Codex CLI)
  --openclaw     Install to ~/.openclaw/skills (OpenClaw)
  --kiro         Install to ~/.kiro/skills (Kiro CLI/IDE)
  --antigravity  Install to ~/.gemini/antigravity/skills (Antigravity IDE)
  --opencode     Install to ~/.agents/skills (OpenCode CLI)
  --adal         Install to ~/.adal/skills (AdaL CLI)
  --path <dir>   Install to <dir> (default: ~/.agent/skills)
  --version <ver>  After clone, checkout tag v<ver> (e.g. 4.6.0 -> v4.6.0)
  --tag <tag>      After clone, checkout this tag (e.g. v4.6.0)

Examples:
  npx agent-skills-hub
  npx agent-skills-hub --cursor
  npx agent-skills-hub --kiro
  npx agent-skills-hub install react-patterns --claude
  npx agent-skills-hub --antigravity
  npx agent-skills-hub --version 4.6.0
  npx agent-skills-hub --path ./my-skills
`);
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (r.status !== 0) process.exit(r.status == null ? 1 : r.status);
}

function main() {
  const opts = parseArgs();
  const { tagArg, versionArg, skillName } = opts;

  if (opts.help) {
    printHelp();
    return;
  }

  const target = defaultDir(opts);
  if (!target || !HOME) {
    console.error('Could not resolve home directory. Use --path <absolute-path>.');
    process.exit(1);
  }

  // If installing a single skill, we need to clone to a temp dir first
  if (skillName) {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-skills-hub-'));
    try {
      console.log(`Cloning to temporary directory to extract skill: ${skillName}…`);
      run('git', ['clone', '--depth', '1', REPO, tempDir], { stdio: 'pipe' });

      // Find the skill in the skills directory recursively
      const skillsDir = path.join(tempDir, 'skills');
      let foundSkillPath = null;

      if (fs.existsSync(skillsDir)) {
        // Simple recursive search
        const search = (dir) => {
          const files = fs.readdirSync(dir);
          for (const file of files) {
            const fullPath = path.join(dir, file);
            const stat = fs.statSync(fullPath);
            if (stat.isDirectory()) {
              if (file === skillName) {
                foundSkillPath = fullPath;
                return;
              }
              search(fullPath);
            }
            if (foundSkillPath) return;
          }
        };
        search(skillsDir);
      }

      if (!foundSkillPath) {
        console.error(`Skill '${skillName}' not found in repository.`);
        process.exit(1);
      }

      const dest = path.join(target, skillName);
      if (fs.existsSync(dest)) {
        console.log(`Skill '${skillName}' already exists at ${dest}. Replacing...`);
        fs.rmSync(dest, { recursive: true, force: true });
      } else {
        // Ensure parent exists
        fs.mkdirSync(path.dirname(dest), { recursive: true });
      }

      fs.cpSync(foundSkillPath, dest, { recursive: true });
      console.log(`Installed skill '${skillName}' to ${dest}`);

      const toolsToInstall = findToolReferences(dest);
      if (toolsToInstall.length > 0) {
        console.log(`Found ${toolsToInstall.length} tool(s) referenced by this skill. Installing...`);
        // If target is like `~/.agent/skills`, tools should go to `~/.agent/tools`
        // If target is `./my-skills`, tools should go to `./my-tools` (or `./tools` depending on preference, we will use sibling approach)
        const targetParentDir = path.dirname(target);
        let toolsTargetDir;
        if (path.basename(target) === 'skills') {
          toolsTargetDir = path.join(targetParentDir, 'tools');
        } else {
          toolsTargetDir = path.join(targetParentDir, target.endsWith('-skills') ? target.replace('-skills', '-tools') : path.basename(target) + '-tools');
        }

        for (const tool of toolsToInstall) {
          const tempToolPath = path.join(tempDir, 'tools', tool);
          const targetToolPath = path.join(toolsTargetDir, tool);

          if (fs.existsSync(tempToolPath)) {
            if (!fs.existsSync(path.dirname(targetToolPath))) {
              fs.mkdirSync(path.dirname(targetToolPath), { recursive: true });
            }
            fs.copyFileSync(tempToolPath, targetToolPath);
            console.log(`  - Installed tool: ${tool}`);
          } else {
            console.warn(`  - Warning: Referenced tool '${tool}' was not found in the repository.`);
          }
        }
      }

    } catch (e) {
      console.error('Error installing skill:', e);
      process.exit(1);
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
    return;
  }

  // Normal full clone/update logic...
  if (fs.existsSync(target)) {
    const gitDir = path.join(target, '.git');
    if (fs.existsSync(gitDir)) {
      console.log('Directory already exists and is a git repo. Updating…');
      process.chdir(target);
      run('git', ['pull']);
      return;
    }
    console.error(`Directory exists and is not a git repo: ${target}`);
    console.error('Remove it or use --path to choose another location.');
    process.exit(1);
  }

  const parent = path.dirname(target);
  if (!fs.existsSync(parent)) {
    try {
      fs.mkdirSync(parent, { recursive: true });
    } catch (e) {
      console.error(`Cannot create parent directory: ${parent}`, e.message);
      process.exit(1);
    }
  }

  if (process.platform === 'win32') {
    run('git', ['-c', 'core.symlinks=true', 'clone', REPO, target]);
  } else {
    run('git', ['clone', REPO, target]);
  }

  const ref = tagArg || (versionArg ? (versionArg.startsWith('v') ? versionArg : `v${versionArg}`) : null);
  if (ref) {
    console.log(`Checking out ${ref}…`);
    process.chdir(target);
    run('git', ['checkout', ref]);
  }

  console.log(`\nInstalled to ${target}`);
  console.log('Pick a bundle in docs/BUNDLES.md and use @skill-name in your AI assistant.');
}

main();
