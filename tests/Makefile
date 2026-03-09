# Put it first so that "make" without argument is like "make help".
PYTEST_OPT := -rf -x -n 2 --dist loadgroup
RUN ?=


test: test_ut  test_it

test_ut:
	mkdir -p ../build/test && cd ../build/test && $(RUN) pytest -x -c ../../tests/pytest.ini $(PYTEST_OPT) \
	../../tests/test_robo_orchard_lab --alluredir allure_unittest/ \
	--cov-config=../../tests/.coveragerc --cov --cov-report=term --cov-report=xml  \
	--cov-report=html --junitxml=unittest.xml --html=unittest.html

test_it:
	mkdir -p ../build/test && cd ../build/test && $(RUN) pytest -x -c ../../tests/pytest.ini -n 1 --dist loadgroup \
	../../tests/test_integration --alluredir allure_integrationtest/ \
	--junitxml=integrationtest.xml --html=integrationtest.html

.PHONY: test
