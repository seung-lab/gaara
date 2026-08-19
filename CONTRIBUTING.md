Contributing
============

Thank you for contributing to this repository! Your contributions can improve the state of the field for everyone.

Before contributing, here are some guidelines to ensure your contribution will be easy to review and a solid improvement to the project.

## Picking Good Improvements

Quality of life improvements are welcome so long as they have a minimal impact on maintainability. For example, if you add automated documentation, it should be configured to be deployed by continuous integration. Additional type annotations should be mechanically validated. If your contribution is difficult to maintain, it may not be accepted or later removed if it causes problems.

Algorithmic improvements are core to the project. However, it's easy to have an idea that is not closely tied to the practical experience of using this algorithm with real data under real conditions. In some of those cases, your idea will be a natural extension, but in other cases it may be off target. It's important to discuss your idea with other practitioners with real experience to gauge its relevance. Ideally, you will have a specific problem you are trying to solve that could be solved with a specific change.

To discuss possible changes, open a discussion or issue thread, or schedule a meeting using the below link. You can also use email for more complex discussions, though often others would benefit from seeing the discussion online.

https://calendly.com/will-silversmith-office-hours/office-hours

## Your Contribution ("Pull Request")

Contributions should be submitted as a "pull request" to the repository on github.com. This means you are submitting your code to be diffed (i.e. compared line-by-line) and reviewed by a team member.

Your changes should be a few hundred lines or ideally less unless you have prior agreement. Your code should come with automated tests if it is a new feature (or a test validating that you fixed a bug). In general, backwards compatibility should be maintained. When necessary, you can ditch clearly unused parameters or deprecate them with a warning for a period of time.

If you need help or get confused, ask questions! ^\_^

## Rules

1. **No significant rewrites.** Do not spring a large rewrite (e.g. thousands of lines of code, change of language) without prior discussion and affirmative agreement. It is very unlikely that this will be accepted even if there are significant advantages to the new system. This code is working and battle tested, many people rely on it, and the maintainer knows the system. It is also often unclear if the contributor will continue to maintain their submission indefinitely. See if you can achieve what you want within the existing structure and ask questions.
2. **LLM generated or assisted code must be of professional human quality.** This is flexible in the sense that disposable one-off-scripts have lower standards than core library routines. You must understand your code and be able to discuss it as if you had written it. If you cannot describe the key decisions and code in your contribution, it will not be accepted.
3. **You must write your own words in documentation.** This is to improve documentation quality and to enhance the audiance's trust in the project. The engineering understanding of the project lives in the human beings that contribute to it and the things they write about it. LLM documentation is often verbose and meandering, and the LLM's context disappeared as soon as it was generated--there's no one to ask questions. Readers should not be expected to invest energy in reading what took no effort to write. Many readers, in part for this reason, have a negative opinion of LLM generated documentation, and this impression reflects on the project as a whole. Minor exceptions can be made, e.g. for validated formatted data tables. However, it is very unlikely that a generative image will be accepted unless it is something like a validated high-quality flow chart without visible artifacts. This is a project used for scientific purposes; generative images can undermine trust in the underlying reality-based nature of the project. It is acceptable to use LLMs to critique what you write, but do not copy-paste their response. LLMs will [ablate the meaning](https://www.theregister.com/software/2026/02/16/semantic-ablation-why-ai-writing-is-boring-and-dangerous/4414930) of your writing and will trip AI text detectors especially now that the EU requires outputs to be [watermarked](https://www.anthropic.com/news/claude-text-watermark).
4. **You must cite your sources.** A citation or link is needed any time you reference, copy, paraphrase, or quote others' work from any medium, e.g. scholarly works, code, books, blogs, etc.

## Setting Up

Create a virtual environment using (for example) the `venv` package, you can also use [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/) (which builds atop `virtualenv`).

### Using `venv`

This will work with Python 3.3+ (as of this writing, we are at Python 3.14) using the standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/seung-lab/gaara.git
# --no-build-isolation required for editable installs using meson
pip install -e . --no-build-isolation
```

### Using `virtualenvwrapper`

This requires a little more setup, but then you get the pleasure of using named virtual environments. Follow the instructions from `virtualenvwrapper` for how to set it up.

```bash
pip install virtualenvwrapper
# ... edit your terminal profile the first time ...
mkvirtualenv -p python3.13 gaara
workon gaara
git clone https://github.com/seung-lab/gaara.git
# --no-build-isolation required for editable installs using meson
pip install -e . --no-build-isolation
```

## Running Tests

Gaara has two ways to run tests, one for Python/Pybind11 and directly in C++ using google-test.

```bash
pip install pytest
python -m pytest -v -x automated_tests.py
```

### Running C++ Tests

You may need to edit the Makefile to set your c++ compiler. These instructions are written for MacOS using homebrew.

```bash
brew install google-test
make test
```

This time profiling profile command (targeting `test.cpp`) will only work on MacOS as it uses xcode instruments.

```bash
make profile
```

