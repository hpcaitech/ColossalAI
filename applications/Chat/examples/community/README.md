# Community Examples
---
We are thrilled to announce the latest updates to ColossalChat, an open-source solution for cloning ChatGPT with a complete RLHF (Reinforcement Learning with Human Feedback) pipeline. 

As Colossal-AI undergoes major updates, we are actively maintaining ColossalChat to stay aligned with the project's progress. With the introduction of Community-driven Pipelines, we aim to create a collaborative platform for developers to contribute exotic features built on top of ColossalChat.

## Community Pipelines

Community-driven Pipelines is an initiative that allows users to contribute their own pipelines to the ColossalChat package, fostering a sense of community and making it easy for others to access and benefit from shared work. The primary goal with community-driven pipelines is to have a community-maintained collection of diverse and exotic functionalities built on top of the ColossalChat package, which is powered by the Colossal-AI project and its Coati module (ColossalAI Talking Intelligence).

For more information about community pipelines, please have a look at this [issue](https://github.com/hpcaitech/ColossalAI/issues/3487).

## Community Examples

Community examples consist of both inference and training examples that have been added by the community. Please have a look at the following table to get an overview of all community examples. Click on the Code Example to get a copy-and-paste ready code example that you can try out. If a community doesn't work as expected, please open an issue and ping the author on it.

| Example                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Code Example                                                      | Colab                                                                                                                                                                                                              |                                                     Author |
|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------:|
| Peft           | Adding Peft support for SFT and Prompts model training                                                                                                                                                                                                                                                                                                                                                                                                                                   | [Huggingface Peft](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples/community/peft)     | - |             [YY Lin](https://github.com/yynil) | 
|...|...|...|...|...|

To load a custom pipeline you just need to pass the `custom_pipeline` argument to `DiffusionPipeline`, as one of the files in `diffusers/examples/community`. Feel free to send a PR with your own pipelines, we will merge them quickly.
```
pipeline = CustomPipeline.from_pretrained("example/colossal-chat-model", custom_pipeline_path="path/to/additional_component")
output = pipeline(input_text)
```

### How to use community-driven features:

Load community-driven pipelines by passing the appropriate argument(s) to the Coati class or any other relevant class, as specified in the examples/community folder. Here's an example of how to use a community-driven pipeline:

```
model_name = "Coati7B"  # Replace with the actual model name
custom_pipeline_name = "CustomPipeline"  # Replace with the filename of the custom pipeline in the 'community_pipelines' folder
chat_instance = ColossalChat.from_pretrained(model_name, custom_pipeline=custom_pipeline_name)
```

Users can load custom pipelines from the `community_pipelines` folder using the code similar to the one you provided. Contribute to the community-driven features by sending a PR with your own implementations to the examples/community folder, and we will merge them promptly.


To share this custom pipeline with the community, you can submit a pull request to add the CustomPipeline code to the `community` folder in the [ColossalChat repository](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples/community). Once the pull request is merged, other users can load the custom pipeline using the code provided in the previous answers.

The general steps can be summarized as:
1. Develop your custom pipeline by inheriting from the `ColossalChat` class and implementing the `__init__` and `__call__` methods as shown in the provided examples.

2. Test your custom pipeline to ensure it works correctly with the ColossalChat framework.

3. Create a new Python file for your custom pipeline and place it in the `community_pipelines` folder.

4. Submit a pull request to add your custom pipeline code to the `community_pipelines` folder in the ColossalChat repository. Please include a brief description of your custom pipeline and its features.

Once your pull request is merged, other users can load your custom pipeline using the `from_pretrained` method in the ColossalChat class. Thank you for contributing to ColossalChat!

### How to get involved
To join our community-driven initiative, please visit the [ColossalChat GitHub repository](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples), review the provided information, and explore the codebase. To contribute, create a new issue outlining your proposed feature or enhancement, and our team will review and provide feedback. We look forward to collaborating with you on this exciting project!
