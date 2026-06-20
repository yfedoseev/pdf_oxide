# Compile the pdf_oxide NIF (c_src/pdf_oxide_nif.c) into priv/pdf_oxide_nif.so.
# Driven by elixir_make (mix compile). Links the default-feature cdylib.
#
# Inputs (override as needed):
#   PDF_OXIDE_INCLUDE_DIR  dir with pdf_oxide_c/pdf_oxide.h   (default ../include)
#   PDF_OXIDE_LIB_DIR      dir with libpdf_oxide.{so,dylib}   (default ../target/release)
PDF_OXIDE_INCLUDE_DIR ?= ../include
PDF_OXIDE_LIB_DIR ?= ../target/release

# ERTS headers (erl_nif.h). erl prints the root; include/ lives under it.
ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~ts/erts-~ts/include", [code:root_dir(), erlang:system_info(version)])' -s init stop)

PRIV = priv
NIF = $(PRIV)/pdf_oxide_nif.so

CFLAGS ?= -O2 -fPIC -Wall -Wextra
CFLAGS += -I$(ERTS_INCLUDE_DIR) -I$(PDF_OXIDE_INCLUDE_DIR)
LDFLAGS += -L$(PDF_OXIDE_LIB_DIR) -lpdf_oxide

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  SO_FLAGS = -dynamiclib -undefined dynamic_lookup
  LDFLAGS += -Wl,-rpath,$(PDF_OXIDE_LIB_DIR)
else
  SO_FLAGS = -shared
  LDFLAGS += -Wl,-rpath,$(PDF_OXIDE_LIB_DIR)
endif

all: $(NIF)

$(NIF): c_src/pdf_oxide_nif.c
	mkdir -p $(PRIV)
	$(CC) $(CFLAGS) $(SO_FLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(NIF)

.PHONY: all clean
