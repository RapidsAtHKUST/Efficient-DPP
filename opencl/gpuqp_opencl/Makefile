#This is the make file for gpuqp_opencl
#created by Bryan Lai from HKUST

HOME_PATH = $(dir $(abspath $(firstword $(MAKEFILE_LIST))))


SRCDIR := $(HOME_PATH)/src
INCDIR := $(HOME_PATH)/inc
BINDIR := $(HOME_PATH)/bin
OBJDIR := $(HOME_PATH)/obj
MAINDIR := $(HOME_PATH)/src/Main
DATADIR := $(HOME_PATH)/data/
COMDIR := $(HOME_PATH)/common


CC := g++ 
CPPFLAGS := -g -O2  -DPROJECT_ROOT=\"$(HOME_PATH)\" -DDATADIR=\"$(DATADIR)\" -DOPENCL_PROJ

PRINT ?= 0
ifeq ($(PRINT), 1)
    CPPFLAGS += -DPRINT_KERNEL
endif

R ?= 0
ifeq ($(R), 1) 
	CPPFLAGS += -DRECORDS
endif

INCLUDE := -I$(INCDIR) \
		   -I$(COMDIR) \
           -I/ghome/zlai/workspace/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common/inc
LINK := -L/ghome/zlai/workspace/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common/lib -lOpenCL

SRC :=  $(filter-out ._%,$(notdir $(shell find -L $(SRCDIR) $(COMDIR) -name *.cpp)))
OBJ := $(SRC:%.cpp=$(OBJDIR)/%.o)
PROGRAM := executor

vpath %.cpp $(SRCDIR)/Join $(SRCDIR)/Utils $(SRCDIR)/Primitive $(SRCDIR)/Main $(SRCDIR)/Test $(COMDIR)
vpath %.o $(OBJDIR)

all: $(PROGRAM)
	
$(PROGRAM):$(OBJ) 
	$(CC) $(CPPFLAGS) -o $(BINDIR)/$@ $^ $(INCLUDE) $(LINK)
$(OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)	
	$(CC) $(CPPFLAGS) -c $< -o $@ $(INCLUDE) $(LINK)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(PROGRAM)
