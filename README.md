# wingbeat-recorder

### Recommended installation method

To install uv on Raspbian:
```wget -qO- https://astral.sh/uv/install.sh | sh```

Assuming that you have ``uv``, ``git``, and ``VS Code`` installed:

```
git clone https://github.com/aubreymoore/wingbeat-recorder
cd wingbeat-recorder

# Create .venv and install dependencies
uv run 

# Open the project in VS Code
code .
```

When you run any of the jupyter notebooks (*.ipynb) for the first time, you will be asked to select a kernel. Select ``.venv/bin/python``.
