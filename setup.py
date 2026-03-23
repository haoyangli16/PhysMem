"""Setup script for physmem package.

Install:
    pip install .                    # Core only (numpy)
    pip install ".[openai]"          # + OpenAI LLM support
    pip install ".[gemini]"          # + Google Gemini support
    pip install ".[llm]"             # + All LLM providers
    pip install ".[faiss]"           # + FAISS similarity search
    pip install ".[clustering]"      # + sklearn clustering + sentence-transformers
    pip install ".[all]"             # Everything
    pip install -e ".[all]"          # Editable install with everything
"""

from setuptools import setup, find_packages

setup(
    name="physmem",
    version="0.1.0",
    description="Physical Memory System for Experience-to-Principle Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["physmem*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "faiss": ["faiss-cpu>=1.7"],
        "faiss-gpu": ["faiss-gpu>=1.7"],
        "openai": ["openai>=1.0.0"],
        "gemini": ["google-genai>=0.1.0"],
        "qwen": ["openai>=1.0.0"],
        "llm": [
            "openai>=1.0.0",
            "google-genai>=0.1.0",
        ],
        "clustering": [
            "scikit-learn>=1.0",
            "sentence-transformers>=2.0",
        ],
        "all": [
            "faiss-cpu>=1.7",
            "openai>=1.0.0",
            "google-genai>=0.1.0",
            "scikit-learn>=1.0",
            "sentence-transformers>=2.0",
        ],
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
    package_data={
        "physmem": ["README.md"],
    },
)
