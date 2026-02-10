const assert = require('assert');
const { hasUseSection } = require('../validate-skills');

const samples = [
  ['## When to Use', true],
  ['## Use this skill when', true],
  ['## When to Use This Skill', true],
  ['## Overview', false],
];

for (const [heading, expected] of samples) {
  const content = `\n${heading}\n- item\n`;
  assert.strictEqual(hasUseSection(content), expected, heading);
}

console.log('ok');
