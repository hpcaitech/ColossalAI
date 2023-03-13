# 📕 Documentation

## 🔗 Table of Contents

- [📕 Documentation](#-documentation)
  - [🔗 Table of Contents](#-table-of-contents)
  - [📝 Overview](#-overview)
  - [🗺 Module Structure](#-module-structure)
  - [🧱 Our Documentation System](#-our-documentation-system)
  - [🎊 Contribution](#-contribution)
    - [🖊 Adding a New Documentation](#-adding-a-new-documentation)
    - [🧹 Doc Testing](#-doc-testing)
    - [💉 Auto Documentation](#-auto-documentation)

## 📝 Overview

We evaluated various existing solutions for documentation in the community and discussed their advantages and disadvantages in the [issue #2651](https://github.com/hpcaitech/ColossalAI/issues/2651). Therefore, we propose to build a more modern and robust documentation system by integrating the Sphinx [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) function and the [Docusaurus](https://docusaurus.io/) framework.

## 🗺 Module Structure

```text
- docs
    - source
        - en
        - zh-Hans
    - sidebars.json
    - versions.json
    - requirements-doc-test.txt
```

The documentation module structure is shown above:
1. source: This folder contains multi-language documentation files.
2. `sidebars.json`: The `sidebars.json` defines the table of content for the tutorials. You need to update this file when a new doc is added/deleted.
3. `versions.json`: The `versions.json` in the **main branch** in the **latest commit** will be used to control the versions to be displayed on our website

## 🧱 Our Documentation System

We believe that the combination of the existing systems can yield several advantages such as simplicity, usability and maintainability:
1. Support [Markdown](https://www.markdownguide.org/). We believe is a more popular language for writing documentation compared to [RST](https://docutils.sourceforge.io/rst.html).
2. Support Autodoc. It can automatically generate documentation from the docstrings in the source code provided by [Sphinx](https://www.sphinx-doc.org/en/master/).
3. Support elegant and modern UI, which is provided by [Docusaurus](https://docusaurus.io/).
4. Support MDX for more flexible and powerful documentation, which is provided by [Docusaurus](https://docusaurus.io/).
5. Support hosting blogs/project home page/other pages besides the documentation, which is provided by [Docusaurus](https://docusaurus.io/).

Therefore, we have built the [ColossalAI-Documentation](https://github.com/hpcaitech/ColossalAI-Documentation) repository to integrate the features above.

## 🎊 Contribution

You can contribute to the documentation by directly setting up a Pull Request towards the `docs/source` folder. There are several guidelines for documentation contribution.

1. The documentation is written in Markdown. You can refer to the [Markdown Guide](https://www.markdownguide.org/) for the syntax.
2. You must ensure that the documentation exists for all languages. You can refer to the [Adding a New Documentation](#-adding-a-new-documentation) for more details.
3. You must provide a test command for your documentation, please see [Doc Testing](#-doc-testing) for more details.
4. You can embed your docstring in your markdown, please see [Auto Documentation](#-auto-documentation) for more details.

### 🖊 Adding a New Documentation

You can add a Markdown file to the `docs/source` folder`. You need to ensure that multi-language is supported in your PR.
Let's assume that you want to add a file called `your_doc.md`, your file structure will look like this.

```text
- docs
  - source
    - en
        - your_doc.md  # written in English
    - zh-Hans
        - your_doc.md  # written in Chinese
  - sidebars.json  # add your documentation file name here
```

Meanwhile, you need to ensure the `sidebars.json` is updated such that it contains your documentation file. Our CI will check whether documentation exists for all languages and can be used to build the website successfully.

### 🧹 Doc Testing

Every documentation is tested to ensure it works well. You need to add the following line to the **bottom of your file** and replace `$command` with the actual command. Do note that the markdown will be converted into a Python file. Assuming you have a `demo.md` file, the test file generated will be `demo.py`. Therefore, you should use `demo.py` in your command, e.g. `python demo.py`.

```markdown
<!-- doc-test-command: $command  -->
```

Meanwhile, only code labeled as a Python code block will be considered for testing.

```markdown
    ```python
    print("hello world")
    ```
```

Lastly, if you want to skip some code, you just need to add the following annotations to tell `docer` to discard the wrapped code for testing.

```markdown
<!--- doc-test-ignore-start -->

    ```python
    print("hello world")
    ```

<!--- doc-test-ignore-end -->
```

If you have any dependency required, please add it to `requriements-doc-test.txt` for pip and `conda-doc-test-deps.yml` for Conda.


### 💉 Auto Documentation

Lastly, you may want to include the API documentation for a class/function in your documentation for reference.
We support `autodoc` to extract the docstring and transform it into a Web element for an elegant display.
You just need to add `{{ autodoc:<mod-name> }}` in your markdown as a single line. An example is given below and you can see the outcome in [this PR](https://github.com/hpcaitech/ColossalAI-Documentation/pull/175).

```markdown
{{ autodoc:colossalai.amp.apex_amp.convert_to_apex_amp }}
```
