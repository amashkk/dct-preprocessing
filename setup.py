"""
DCT 預處理套件安裝腳本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dct-preprocessing",
    version="1.0.0",
    author="孔祥庭, 王翊丞",
    author_email="",
    description="用於減少振鈴失真的 DCT 影像壓縮選擇性預處理方法",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dct-preprocessing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dct-preprocess=examples.demo:main",
        ],
    },
)
