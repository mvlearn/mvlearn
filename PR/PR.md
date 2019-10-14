# PR checklist

0. Make sure you download any new packages added to repo by running `pip3 install -r requirements.txt`

1. Code added to `multiview` should be in a very specific folder
  (e.g. If it's an embedding algorithm: `multiview/embedding`)

2. All directories should have an empty `__init__.py` script

3. All functions should have comments following napoleon formatting (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/). See `multiview/example` for the correct format.

4. Unit tests for code should be written in the tests/ folder in the root directory using pytest (https://docs.pytest.org/en/latest/contents.html). Unit tests should be thorough (all main and helper functions should have tests).

5. Make sure your tests all pass by running `pytest tests/` from the root directory

6. Make sure your test has correct style by running `pycodestyle multiview/`

7. If you added any new packages, include it by running `pip3 freeze >requirements.txt` (this is only if you are using a virtual environment. If not, download `pipreqs` and use that instead). If you add too many unnecessary packages your PR will not be approved.

8. Create the PR and make sure the build passes and test coverage does not drop.

9. Make sure to squash commits after your PR is reviewed and ready to be merged in.
