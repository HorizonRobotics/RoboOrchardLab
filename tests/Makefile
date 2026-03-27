# Put it first so that "make" without argument is like "make help".
UT_RUNNER := ../../tests/run_unit_tests.py
PYTEST_RC := ../../tests/pytest.ini
UT_ALLURE_DIR := allure_unittest
UT_COVERAGE_FILE := .coverage.ut
UT_NON_SIM_WORKERS := 8
UT_SIM_WORKERS := 4
RUN ?=


test: test_ut  test_it

test_ut:
	mkdir -p ../build/test && cd ../build/test && $(RUN) python3 $(UT_RUNNER) \
	--non-sim-workers $(UT_NON_SIM_WORKERS) \
	--sim-workers $(UT_SIM_WORKERS) \
	--allure-dir $(UT_ALLURE_DIR) \
	--coverage-file $(UT_COVERAGE_FILE)

test_it:
	mkdir -p ../build/test && cd ../build/test && $(RUN) pytest -x -c $(PYTEST_RC) -n 1 --dist loadgroup \
	../../tests/test_integration --alluredir allure_integrationtest/ \
	--junitxml=integrationtest.xml --html=integrationtest.html

.PHONY: test
