## Example pull request workflow criteria

1. Overview
To automatically, periodically check the correctness of the example code, the github action is used to examine the code as the implementation of CI/CD(Continuous integration/continuous deployment) concept. This doc is to standardrize the example format. Nonstandard pull request(PR) will be blocked automatically. 
2. Testing Design
Several code formats are listed here.
  A. The environment variable should be written in the file, not separately set in the terminal.
  B. The dependent libraries should be listed in requirements.txt and add the command "pip install -r requirements.txt" in the test_ci.sh file. 
  C. The config parameters should be small for fast testing
  D. Add one test_ci.sh file to start the training. 
3. File Structure Chart

       └─examples
          └─images
              └─vit
                └─requirements.txt
                └─test_ci.sh
