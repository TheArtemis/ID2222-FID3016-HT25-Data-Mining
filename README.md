# ID2222-FID3016-HT25-Data-Mining

A Python project for data mining built with [uv](https://docs.astral.sh/uv/).

## Prerequisites

Install uv if you haven't already:

```bash
pip install uv
```

## Getting Started

### Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies.

### Run the Application

```bash
uv run main.py
```

### Add New Dependencies

To add a regular dependency:

```bash
uv add <package-name>
```

To add a development dependency:

```bash
uv add --dev <package-name>
```

### Python Version

This project requires Python 3.12 or higher. The specific version is pinned in `.python-version`.