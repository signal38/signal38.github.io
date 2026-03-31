PORT ?= 8080

.PHONY: preview
preview:
	@echo ""
	@echo "\033[1;32m  http://localhost:$(PORT)\033[0m"
	@echo ""
	@python3 -m http.server $(PORT) --directory docs
