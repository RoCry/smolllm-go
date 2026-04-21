SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

include hack/common.mk

CPUS ?= $(shell (nproc --all || sysctl -n hw.ncpu) 2>/dev/null || echo 1)
MAKEFLAGS += --warn-undefined-variables -j$(CPUS) --no-print-directory

all: help

##@ General

# The help target lists documented goals (## comments) grouped by category (##@).
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

BIN_DIR := $(CURDIR)/bin
GOLANGCI_LINT_VERSION ?= v2.11.4
GOLANGCI_LINT_BIN_DIR := $(BIN_DIR)/golangci-lint-$(GOLANGCI_LINT_VERSION)
GOLANGCI_LINT := $(GOLANGCI_LINT_BIN_DIR)/golangci-lint

##@ Tooling

lint: $(GOLANGCI_LINT) ## Run golangci-lint across the module.
	$(GOLANGCI_LINT) run --config=.golangci.yml ./...

golangci-lint: $(GOLANGCI_LINT) ## Install golangci-lint.

$(GOLANGCI_LINT):
	@mkdir -p $(GOLANGCI_LINT_BIN_DIR)
	$(call go-get-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION))

clean-lint: ## Remove the pinned golangci-lint binary.
	rm -rf $(GOLANGCI_LINT_BIN_DIR)

##@ Tests

GO_TEST_FLAGS ?= -race

test: ## Run Go test suite with the race detector.
	go test -v $(GO_TEST_FLAGS) ./...

.PHONY: all help lint golangci-lint clean-lint test
