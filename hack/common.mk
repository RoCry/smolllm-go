define run-in-folder
[ -d $(1) ] && { \
	set -e ;\
	echo "Running $(2) in $(1)"; \
	cd $(1); \
	$(2); \
	cd - > /dev/null; \
}
endef

# go-get-tool will 'go get' any package $2 and install it to $1.
PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
define go-get-tool
@[ -f $(1) ] || { \
	set -e ;\
	TMP_DIR=$$(mktemp -d) ;\
	cd $$TMP_DIR ;\
	go mod init tmp ;\
	echo "Downloading $(2)" ;\
	GOBIN=$(shell dirname $(1)) go install $(2) ;\
	rm -rf $$TMP_DIR ;\
}
endef
