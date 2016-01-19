#This is the make file for gpuqp_cuda
#created by Bryan Lai from HKUST

HOME_PATH = $(shell pwd)

SRCDIR := $(HOME_PATH)/src
INCDIR := $(HOME_PATH)/inc
BINDIR := $(HOME_PATH)/bin
OBJDIR := $(HOME_PATH)/obj
KERNELDIR := $(HOME_PATH)/kernels

NVCC := nvcc
CC := gcc

CC_FLAGS := -g  -DPROJECT_ROOT=\"$(HOME_PATH)\"
NVCC_FLAGS := -g

PRINT ?= 0
ifeq ($(PRINT), 1)
    CC_FLAGS += -DPRINT_KERNEL
endif

INCLUDE := 	-I$(INCDIR) \
			-I/usr/local/cuda/samples/common/inc \
           	-I/usr/local/cuda/include
LINK := -L /usr/local/cuda/lib64 -lcudart -lcuda

CPP_SRC :=  $(filter-out ._%,$(notdir $(shell find src -name *.cpp)))
CU_SRC :=  $(filter-out ._%,$(notdir $(shell find kernels -name *.cu)))
CPP_OBJ := $(CPP_SRC:%.cpp=$(OBJDIR)/%.o)
CU_OBJ := $(CU_SRC:%.cu=$(OBJDIR)/%.o)

PROGRAM := gpuqp_cuda

vpath %.cpp $(SRCDIR) 
vpath %.cu $(KERNELDIR)
vpath %.o $(OBJDIR)

all: $(PROGRAM)
	 
$(PROGRAM):$(CPP_OBJ) $(CU_OBJ) 
	$(NVCC) -o $(BINDIR)/$@ $^ $(LINK)
$(CPP_OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)	
	$(CC) $(CC_FLAGS) -c $< -o $@ $(INCLUDE)

$(CU_OBJ): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(OBJDIR)	
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(INCLUDE)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(PROGRAM)
