from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="datamigrationapp",
    version="0.1.0",
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=requirements,
    author="ak",
    author_email="admin@example.com",
    description="A comprehensive data migration tool with React and Streamlit frontends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="data migration, flask, react, streamlit, mlops",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        'console_scripts': [
            'datamigrationapp=backend.app:main',
        ],
    },
) 