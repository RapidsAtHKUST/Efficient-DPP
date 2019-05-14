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
CPPFLAGS := -g -O3 -DPROJECT_ROOT=\"$(HOME_PATH)\" -Wno-write-strings

INCLUDE := -I$(INCDIR) \
		   -I$(COMDIR) 
LINK := -L/opt/intel/opencl/lib64 -lOpenCL
#LINK := -L/usr/local/cuda/lib64/ -lOpenCL

SRC :=  $(filter-out ._%,$(notdir $(shell find -L $(SRCDIR) $(COMDIR) -name *.cpp)))
OBJ := $(SRC:%.cpp=$(OBJDIR)/%.o)
PROGRAM := executor

vpath %.cpp $(SRCDIR)/Utils $(SRCDIR)/Impl $(SRCDIR)/Main $(SRCDIR)/Test $(COMDIR)
vpath %.o $(OBJDIR)

all: $(PROGRAM)

sinclude $(SRC:%.cpp=$(OBJDIR)/%.d)

$(PROGRAM):$(OBJ) 
	$(CC) $(CPPFLAGS) -o $(BINDIR)/$@ $^ $(INCLUDE) $(LINK)
$(OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)	
	$(CC) -MM -MP -MT $@ -MF $(@:.o=.d) $< $(INCLUDE)
	$(CC) $(CPPFLAGS) -c $< -o $@ $(INCLUDE) $(LINK)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)/* $(BINDIR)/$(PROGRAM)
