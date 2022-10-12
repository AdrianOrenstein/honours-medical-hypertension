PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	@bash scripts/build_docker.bash

run:
	@bash scripts/run.bash

stop:
	@docker container kill $$(docker ps -q)

jupyter:
	@bash scripts/jupyter.bash

lint:
	@bash scripts/lint.bash
	@echo "✅✅✅✅✅ Lint is good! ✅✅✅✅✅"

test:
	@bash scripts/tests.bash
	@echo "✅✅✅✅✅ Tests are good! ✅✅✅✅✅"

