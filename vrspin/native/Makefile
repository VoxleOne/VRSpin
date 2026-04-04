# VRSpin Native Library — Build Instructions
#
# Builds libvrspin_native.so (Linux/macOS) or vrspin_native.dll (Windows).
# No external dependencies beyond a C99 compiler and libm.
#
# Usage:
#   make            # build shared library
#   make clean      # remove build artifacts
#   make test       # build and run the ctypes test via pytest

CC      ?= gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Wpedantic -fPIC -DVRSPIN_BUILD_DLL
LDFLAGS := -shared -lm

SRCDIR  := .
SRC     := $(SRCDIR)/vrspin_native.c
HDR     := $(SRCDIR)/vrspin_native.h

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  TARGET := libvrspin_native.dylib
else ifeq ($(OS),Windows_NT)
  TARGET := vrspin_native.dll
else
  TARGET := libvrspin_native.so
endif

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f libvrspin_native.so libvrspin_native.dylib vrspin_native.dll

test: $(TARGET)
	cd ../.. && python -m pytest tests/test_native_bridge.py -v
