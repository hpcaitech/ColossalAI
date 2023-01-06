## Examples folder document

## Table of Contents
<ul>
 <li><a href="#Example-folder-description">Example folder description</a> </li>
 <li><a href="#Integrate-Your-Example-With-System-Testing">Integrate Your Example With System Testing</a> </li>
</ul>

## Example folder description

This folder provides several examples using colossalai. The images folder includes model like diffusion, dreambooth and vit. The language folder includes gpt, opt, palm and roberta. The tutorial folder is for concept illustration, such as auto-parallel, hybrid-parallel and so on.


## Integrate Your Example With System Testing

For example code contributor, to meet the expectation and test your code automatically using github workflow function, here are several steps:


- (must) Have a test_ci.sh file in the folder like shown below in 'File Structure Chart'
- The dataset should be located in the company's machine and can be announced using environment variable and thus no need for a separate terminal command.
- The model parameters should be small to allow fast testing.
- File Structure Chart

       └─examples
          └─images
              └─vit
                └─requirements.txt
                └─test_ci.sh
