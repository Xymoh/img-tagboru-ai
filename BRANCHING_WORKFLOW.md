# Branching Workflow for img-tagger

## Protected Main Branch
- **main**: Production releases only
  - Tagged releases (v1.0.0, etc.)
  - Merged from develop via PR
  - No direct commits allowed

## Development Branches

### develop
- Integration branch for features
- Always has latest working code
- Created from: `git checkout -b develop origin/main`
- Merges back to main via PR when ready for release

### Feature branches
- For new features or bug fixes
- Name: `feature/description` or `bugfix/issue-name`
- Created from: `git checkout -b feature/my-feature develop`
- Example: `git checkout -b feature/add-gpu-support develop`

## Workflow

```bash
# 1. Get latest develop
git fetch origin develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/my-feature develop

# 3. Make changes and commit (NO DIRECT MAIN COMMITS)
git add .
git commit -m "feat: description"
git push origin feature/my-feature

# 4. Create Pull Request on GitHub
#    - Set base: develop
#    - Set head: feature/my-feature
#    - Wait for review/merge

# 5. When develop is ready for release:
#    - Create PR: develop → main
#    - After merge, tag: git tag v1.1.0
#    - Push tag: git push origin v1.1.0
#    - GitHub Actions builds from tag automatically
```

## Current Configuration

- **windows-build.yml**: Only triggers on `v*` tags or manual `workflow_dispatch`
- No auto-builds on push to main (requires tags)
- Build artifacts available after tag/release

## Important Rules

✅ DO:
- Work on feature branches
- Create PRs into develop
- Tag releases (v1.0.0, v1.1.0)
- Use develop branch as integration point

❌ DON'T:
- Push directly to main
- Force push to develop
- Skip PRs
- Tag commits that aren't on main

---

Current branches:
- main (protected) → production releases
- develop → active development
