# Getting Started: Your First Contribution

This tutorial will guide you through making your first contribution to the project, from setting up your development environment to submitting a pull request.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Set up a local development environment
- Understand the project structure
- Make a small code change
- Test your changes
- Submit a pull request

## Prerequisites

- Basic understanding of Git and version control
- Familiarity with Python and/or JavaScript
- A GitHub account

## Step 1: Set Up Your Development Environment

First, let's get your local development environment ready. Follow the [Development Environment Setup Guide](../how_to_guides/development_environment_setup.md) to install all necessary dependencies.

**Quick verification:**
```bash
# Check if you can run the project
python manage.py --version
# or
npm --version
```

## Step 2: Clone and Explore the Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/your-project.git
   cd your-project
   ```

2. **Explore the project structure:**
   ```
   ├── Documentation/          # Project documentation
   ├── src/                    # Source code
   ├── tests/                  # Test files
   ├── requirements.txt        # Python dependencies
   ├── package.json            # Node.js dependencies
   └── README.md              # Project overview
   ```

3. **Check the main documentation:**
   - Read the [Project Overview](../01_Project_Overview/README.md)
   - Review [Coding Standards](../10_Standards_and_Best_Practices/coding_standards.md)

## Step 3: Create a Feature Branch

Always create a new branch for your changes:

```bash
# Create and switch to a new branch
git checkout -b feature/my-first-contribution

# Verify you're on the new branch
git branch
```

## Step 4: Make a Small Change

Let's start with a simple documentation improvement:

1. **Find a file to improve:**
   - Look for TODO comments: `grep -r "TODO" Documentation/`
   - Check for incomplete sections in README files
   - Look for placeholder text like "[Enter description here]"

2. **Make your change:**
   - Choose a simple improvement (fixing a typo, completing a TODO)
   - Make sure your change is meaningful but small

3. **Example change:**
   Let's say you found a TODO in a README file. Change:
   ```
   ## Features
   TODO: List project features
   ```
   To:
   ```
   ## Features
   - User authentication and authorization
   - RESTful API endpoints
   - Database integration
   - Documentation generation
   ```

## Step 5: Test Your Changes

1. **Run relevant tests:**
   ```bash
   # Run all tests
   python manage.py test
   # or
   npm test

   # Run specific tests if available
   python manage.py test tests.test_specific_feature
   ```

2. **Verify documentation builds:**
   ```bash
   # If using MkDocs
   mkdocs build

   # Or check markdown syntax
   find Documentation/ -name "*.md" -exec markdownlint {} \;
   ```

## Step 6: Commit Your Changes

1. **Check what files changed:**
   ```bash
   git status
   ```

2. **Stage and commit your changes:**
   ```bash
   # Stage all changes
   git add .

   # Commit with a descriptive message
   git commit -m "docs: Complete features section in README

   - Add list of main project features
   - Remove TODO placeholder
   - Improve documentation completeness"
   ```

## Step 7: Push and Create Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/my-first-contribution
   ```

2. **Create a Pull Request:**
   - Go to GitHub and navigate to your repository
   - Click "Compare & pull request"
   - Fill out the PR template:
     - **Title**: Brief description of your change
     - **Description**: Explain what you changed and why
     - **Related Issues**: Link any related issues

3. **PR Description Example:**
   ```
   ## Description
   This PR completes the features section in the main README by replacing the TODO placeholder with an actual list of project features.

   ## Changes Made
   - Added bullet points describing main project features
   - Removed TODO placeholder
   - Improved overall documentation quality

   ## Testing
   - Verified markdown syntax is correct
   - Confirmed documentation builds successfully
   - No functional changes to code
   ```

## Step 8: Address Feedback

1. **Monitor your PR:**
   - Check for comments from maintainers
   - Address any requested changes
   - Push additional commits if needed

2. **Respond to feedback:**
   ```bash
   # Make additional changes
   git add .
   git commit -m "Address PR feedback: fix formatting"
   git push origin feature/my-first-contribution
   ```

## Success!

Congratulations! You've completed your first contribution. 🎉

## What You Learned

- How to set up a development environment
- Project structure and organization
- Git workflow for contributions
- Documentation improvement process
- Pull request creation and management

## Next Steps

- Look for more substantial contributions
- Join project discussions
- Help review other contributors' PRs
- Explore the [Development Guides](../06_Development_Guides/) for advanced topics

## Troubleshooting

### Issue: Tests are failing
**Solution:** Check the error messages and fix any issues. If unsure, ask for help in your PR comments.

### Issue: Documentation doesn't build
**Solution:** Verify your markdown syntax and check for broken links.

### Issue: PR has merge conflicts
**Solution:**
```bash
# Update your main branch
git checkout main
git pull origin main

# Rebase your branch
git checkout feature/my-first-contribution
git rebase main

# Resolve any conflicts and push
git push origin feature/my-first-contribution --force-with-lease
```

## Getting Help

- Check the [Development Guides](../06_Development_Guides/)
- Ask questions in your PR comments
- Join team communication channels
- Review the [Contributing Guidelines](../06_Development_Guides/README.md)