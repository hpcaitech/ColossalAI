## Integrate Your Example With System Testing

- Overview

To automatically, periodically check the correctness of the example code, the GitHub action is used to examine the code as the implementation of CI/CD(Continuous integration/continuous deployment) concept. This doc is to standardize the example format. Nonstandard pull request(PR) will be blocked automatically. 


- Testing Design

The code submitted will be checked by the following three ways: the changed code within examples folder will be checked; all codes within examples folder will be checked each Sunday (Singapore time) 00:00; one can manually assert a sub-folder in examples folder to check. 

All settings should be listed in test_ci.sh file. Some reminders are shown below.

  A. The environment variable should be written in the file, not separately set in the terminal.

  B. The dependent libraries can be listed in requirements.txt and add the command "pip install -r requirements.txt" in the test_ci.sh file. 

  C. The config parameters should be small for fast testing

  D. Add one test_ci.sh file for testing. 


- File Structure Chart

       └─examples
          └─images
              └─vit
                └─requirements.txt
                └─test_ci.sh
