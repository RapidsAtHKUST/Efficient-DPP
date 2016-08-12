#This is the make file for gpuqp_openmp
#created by Bryan Lai from HKUST

HOME_PATH = $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

SRCDIR := $(HOME_PATH)/src
INCDIR := $(HOME_PATH)/inc
BINDIR := $(HOME_PATH)/bin
OBJDIR := $(HOME_PATH)/obj
COMDIR := $(HOME_PATH)/common

CC := icc

CC_FLAGS := -O3 -DPROJECT_ROOT=\"$(HOME_PATH)\" -DOPENMP_PROJ -fopenmp -qopt-report=3 -qopt-report-phase=vec #-qopt-streaming-stores always

PRINT ?= 0
ifeq ($(PRINT), 1)
    CC_FLAGS += -DPRINT_KERNEL
endif

INCLUDE := 	-I$(INCDIR) \
			-I$(COMDIR) \
			-I/ghome/zlai/intel/tbb/include

LINK :=  -tbb

CPP_SRC :=  $(filter-out ._%,$(notdir $(shell find -L $(SRCDIR) $(COMDIR) -name *.cpp)))
CPP_OBJ := $(CPP_SRC:%.cpp=$(OBJDIR)/%.o)


PROGRAM := executor

vpath %.cpp $(SRCDIR) $(COMDIR)
vpath %.o $(OBJDIR)

all: $(PROGRAM)

sinclude $(CPP_SRC:%.cpp=$(OBJDIR)/%.d)

$(PROGRAM):$(CPP_OBJ)  
	$(CC) -o $(BINDIR)/$@ $^ $(LINK)
$(CPP_OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)	
	$(CC) -MM -MP -MT $@ -MF $(@:.o=.d) $< $(INCLUDE)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(INCLUDE)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(PROGRAM)
