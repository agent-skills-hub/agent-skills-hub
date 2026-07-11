const assert = require('node:assert/strict');

const { chooseGeneratedAt } = require('../build-catalog');

const skills = [
  {
    id: 'example',
    name: 'example',
    description: 'Example skill',
    category: 'development',
    tags: ['example'],
    triggers: ['example'],
    path: 'skills/example/SKILL.md',
  },
];

const previousTimestamp = '2026-01-01T00:00:00.000Z';
const currentTimestamp = '2026-02-01T00:00:00.000Z';
const previousCatalog = {
  generatedAt: previousTimestamp,
  total: skills.length,
  skills,
};

assert.equal(chooseGeneratedAt(previousCatalog, skills, currentTimestamp), previousTimestamp);
assert.equal(chooseGeneratedAt(previousCatalog, [...skills, { ...skills[0], id: 'new' }], currentTimestamp), currentTimestamp);
assert.equal(chooseGeneratedAt(null, skills, currentTimestamp), currentTimestamp);

process.stdout.write('ok\n');
