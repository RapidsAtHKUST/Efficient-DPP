#This is the make file for gpuqp_cuda
#created by Zhuohang Lai Lai from HKUST

HOME_PATH = $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

SRCDIR := $(HOME_PATH)/src
INCDIR := $(HOME_PATH)/inc
BINDIR := $(HOME_PATH)/bin
OBJDIR := $(HOME_PATH)/obj
KERNELDIR := $(HOME_PATH)/kernels
COMDIR := $(HOME_PATH)/common

NVCC := nvcc
CC := g++

CC_FLAGS := -O3 -DPROJECT_ROOT=\"$(HOME_PATH)\" -DCUDA_PROJ
NVCC_FLAGS := -O3 -Xptxas -dlcm=ca -DCUDA_PROJ

PRINT ?= 0
R ?= 0

ifeq ($(PRINT), 1)
    CC_FLAGS += -DPRINT_KERNEL
endif

ifeq ($(R), 1)
    CC_FLAGS += -DRECORDS
	NVCC_FLAGS += -DRECORDS
endif

INCLUDE := 	-I$(INCDIR) \
			-I$(COMDIR) \
			-I/usr/local/cuda/samples/common/inc \
           	-I/usr/local/cuda/include
LINK := -L /usr/local/cuda/lib64 -lcudart -lcuda

CPP_SRC :=  $(filter-out ._%,$(notdir $(shell find -L $(SRCDIR) $(COMDIR) -name *.cpp)))
CU_SRC :=  $(filter-out ._%,$(notdir $(shell find -L $(KERNELDIR) -name *.cu)))

CPP_OBJ := $(CPP_SRC:%.cpp=$(OBJDIR)/%.o)
CU_OBJ := $(CU_SRC:%.cu=$(OBJDIR)/%.o)

PROGRAM := executor

vpath %.cpp $(SRCDIR) $(COMDIR)
vpath %.cu $(KERNELDIR)
vpath %.o $(OBJDIR)

all: $(PROGRAM)

$(PROGRAM):$(CPP_OBJ) $(CU_OBJ) 
	$(CC) -o $(BINDIR)/$@ $^ $(LINK)

sinclude $(CPP_SRC:%.cpp=$(OBJDIR)/%.d)
sinclude $(CU_SRC:%.cu=$(OBJDIR)/%.du)

$(CPP_OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)	
	$(CC) -MM -MP -MT $@ -MF $(@:.o=.d) $< $(INCLUDE)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(INCLUDE)

$(CU_OBJ): $(OBJDIR)/%.o: %.cu 
	@mkdir -p $(OBJDIR)	
	$(NVCC) -E -Xcompiler "-MM -MP -MT $@ -MF $(@:.o=.du)" $< $(INCLUDE)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(INCLUDE)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)/* $(BINDIR)/$(PROGRAM)





