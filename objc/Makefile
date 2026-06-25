# pdf_oxide — Objective-C binding build (macOS, clang).
#
# Links the default-feature cdylib. Override paths if your layout differs:
#   make PDF_OXIDE_INCLUDE_DIR=... PDF_OXIDE_LIB_DIR=...
PDF_OXIDE_INCLUDE_DIR ?= ../include
PDF_OXIDE_LIB_DIR ?= ../target/release

CC ?= clang
CFLAGS = -fobjc-arc -Wall -Wextra -Iinclude -I$(PDF_OXIDE_INCLUDE_DIR)
LDFLAGS = -L$(PDF_OXIDE_LIB_DIR) -lpdf_oxide -framework Foundation \
          -Wl,-rpath,$(PDF_OXIDE_LIB_DIR)
SRC = src/POXPdfOxide.m

build: test example

test: $(SRC) tests/test_api_coverage.m
	$(CC) $(CFLAGS) $(SRC) tests/test_api_coverage.m $(LDFLAGS) -o test_api_coverage

example: basic_extraction html_extraction words_geometry tables_extraction

basic_extraction: $(SRC) examples/basic_extraction.m
	$(CC) $(CFLAGS) $(SRC) examples/basic_extraction.m $(LDFLAGS) -o basic_extraction

html_extraction: $(SRC) examples/html_extraction.m
	$(CC) $(CFLAGS) $(SRC) examples/html_extraction.m $(LDFLAGS) -o html_extraction

words_geometry: $(SRC) examples/words_geometry.m
	$(CC) $(CFLAGS) $(SRC) examples/words_geometry.m $(LDFLAGS) -o words_geometry

tables_extraction: $(SRC) examples/tables_extraction.m
	$(CC) $(CFLAGS) $(SRC) examples/tables_extraction.m $(LDFLAGS) -o tables_extraction

check: test
	DYLD_LIBRARY_PATH=$(PDF_OXIDE_LIB_DIR) ./test_api_coverage

clean:
	rm -f test_api_coverage basic_extraction html_extraction words_geometry tables_extraction

.PHONY: build test example check clean
