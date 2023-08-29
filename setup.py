import setuptools

setuptools.setup(
    name="skit",
    version="0.1.0",
    author="Shadi Hamdan et al.",
    author_email="shamdan17@ku.edu.tr",
    description="Shadi's Toolkit",
    long_description="A collection of useful functions and toolkits I use in my projects.",
    long_description_content_type="text",
    url="https://github.com/Shamdan17/skit",
    project_urls={
        "Bug Tracker": "https://github.com/Shamdan17/skit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
)
