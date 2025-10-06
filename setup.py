#!/usr/bin/env python3
"""
Setup script for Medical Imaging AI API
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-imaging-ai-api",
    version="1.0.0",
    author="Medical Imaging AI Research Team",
    author_email="contact@medicalimagingai.com",
    description="A comprehensive framework for automated medical image analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Web8080/Medical_Imaging_AI_API_Research_Paper_and_web_app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-imaging-api=src.api.main:main",
            "medical-imaging-dashboard=frontend.streamlit_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="medical imaging, ai, machine learning, deep learning, healthcare, api, fastapi, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/Web8080/Medical_Imaging_AI_API_Research_Paper_and_web_app/issues",
        "Source": "https://github.com/Web8080/Medical_Imaging_AI_API_Research_Paper_and_web_app",
        "Documentation": "https://github.com/Web8080/Medical_Imaging_AI_API_Research_Paper_and_web_app/tree/main/docs",
    },
)
