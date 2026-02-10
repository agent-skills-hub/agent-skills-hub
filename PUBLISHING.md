# Publishing to NPM

This guide describes how to publish `agent-skills-hub` to NPM.

## Prerequisites

1.  **NPM Account**: You must have an account on [npmjs.com](https://www.npmjs.com/).
2.  **Login**: Ensure you are logged in to npm via the terminal.
    ```bash
    npm login
    ```

## Publishing Steps

1.  **Preparation**: Ensure `package.json` has the correct `name`, `version`, and `repository` fields.

2.  **Versioning**: Update the version number. semantic versioning is recommended (Major.Minor.Patch).

    ```bash
    npm version patch # or minor, or major
    ```

3.  **Publish**: Run the publish command.

    ```bash
    npm publish --access public
    ```

    _Note: The `--access public` flag is required for scoped packages or the first publication of a new package to ensure it's publicly available._

4.  **Verification**: Check [npmjs.com/package/agent-skills-hub](https://www.npmjs.com/package/agent-skills-hub) to verify the new version is live.

## Automation (Optional)

You can automate this process using GitHub Actions by setting up a workflow that runs `npm publish` on release creation. Ensure you add `NPM_TOKEN` to your repository secrets.
